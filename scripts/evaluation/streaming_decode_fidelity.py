from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from text2motion.app.config import load_config
from text2motion.app.run_log import log_metrics, start_run
from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM, FPS
from text2motion.motion.storage import FEATURES_DIR, SPLIT_FILES
from text2motion.streaming.decoder import StreamingMotionDecoder, measure_decoder_context
from text2motion.tokenization.model import build_tokenizer_network


def load_tokenizer(cfg, ckpt_path: str):
    tokenizer = build_tokenizer_network(cfg.tokenizer, cfg.rvq_baseline)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "tokenizer"):
            if key in state:
                state = state[key]
                break
    tokenizer.load_state_dict(state)
    return tokenizer.eval()


def load_clips(
    root: Path, split: Split, max_clips: int, window: int
) -> tuple[np.ndarray, MotionScaler]:
    scaler = MotionScaler.load(root, dim=DIM)
    ids = [
        line.strip()
        for line in (root / SPLIT_FILES[split]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clips = []
    for clip_id in ids:
        base = clip_id[1:] if clip_id.startswith("M") else clip_id
        path = root / FEATURES_DIR / f"{base}.npy"
        if not path.is_file():
            continue
        feats = np.load(path)
        if feats.shape[0] < window:
            continue
        clips.append(scaler.normalize(feats[:window]))
        if len(clips) >= max_clips:
            break
    if not clips:
        raise RuntimeError(f"no clips of at least {window} frames found in {root}/{split.value}")
    return np.stack(clips), scaler


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg, args.ckpt)
    if cfg.paths.hml3d_out_dir is None:
        raise ValueError("paths.hml3d_out_dir must be configured")
    window = args.window or cfg.data.max_motion_len
    fps = args.fps or FPS
    clips, scaler = load_clips(
        Path(cfg.paths.hml3d_out_dir), args.split, args.max_clips, window
    )
    motion = torch.from_numpy(clips).float()
    mean_t = torch.from_numpy(scaler.mean).float()
    std_t = torch.from_numpy(scaler.std).float()

    measured_left, measured_right = measure_decoder_context(
        tokenizer, cfg.tokenizer.downsample
    )
    print(f"measured decoder context: left {measured_left} tokens, right {measured_right} tokens")

    run_dir = start_run("streaming_decode_fidelity", cfg, cfg.paths.logs_dir)

    with torch.no_grad():
        indices = tokenizer.encode(motion)
        reference = tokenizer.decode(indices) * std_t + mean_t
    scale = reference.abs().mean()
    token_list = [indices[:, step] for step in range(indices.shape[1])]

    rows = []
    for lookahead in range(args.max_lookahead + 1):
        decoder = StreamingMotionDecoder(
            tokenizer,
            cfg.tokenizer.downsample,
            mean_t,
            std_t,
            chunk_tokens=args.chunk_tokens,
            left_context=measured_left,
            lookahead=lookahead,
        )
        streamed = torch.cat(list(decoder.stream_tokens(iter(token_list))), dim=1)
        diff = (streamed - reference).abs()
        row = {
            "lookahead_tokens": lookahead,
            "lookahead_frames": lookahead * cfg.tokenizer.downsample,
            "latency_seconds": lookahead * cfg.tokenizer.downsample / fps,
            "state_tokens": decoder.state_tokens,
            "mae": float(diff.mean()),
            "max_abs": float(diff.max()),
            "relative_percent": float(100 * diff.mean() / scale),
        }
        rows.append(row)
        print(
            f"lookahead {lookahead} ({row['lookahead_frames']:3d} frames, "
            f"{row['latency_seconds']:.2f}s)  state {row['state_tokens']:2d} tok  "
            f"MAE {row['mae']:.7f}  max {row['max_abs']:.5f}  rel {row['relative_percent']:7.4f}%"
        )

    no_context = []
    for chunk in args.chunk_grid:
        decoder = StreamingMotionDecoder(
            tokenizer,
            cfg.tokenizer.downsample,
            mean_t,
            std_t,
            chunk_tokens=chunk,
            left_context=0,
            lookahead=0,
        )
        streamed = torch.cat(list(decoder.stream_tokens(iter(token_list))), dim=1)
        diff = (streamed - reference).abs()
        entry = {
            "chunk_tokens": chunk,
            "chunk_frames": chunk * cfg.tokenizer.downsample,
            "relative_percent": float(100 * diff.mean() / scale),
            "max_abs": float(diff.max()),
        }
        no_context.append(entry)
        print(
            f"[no context] chunk {chunk:3d} ({entry['chunk_frames']:3d} frames)  "
            f"rel {entry['relative_percent']:6.2f}%  max {entry['max_abs']:.4f}"
        )

    log_metrics(
        run_dir,
        {
            "checkpoint": args.ckpt,
            "split": args.split.value,
            "clips": int(motion.shape[0]),
            "window_frames": window,
            "chunk_tokens": args.chunk_tokens,
            "downsample": cfg.tokenizer.downsample,
            "measured_left_context": measured_left,
            "measured_right_context": measured_right,
            "lookahead_sweep": rows,
            "no_context_chunk_sweep": no_context,
        },
    )
    print(f"run dir: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer/fsq_g8_v1024.yaml")
    parser.add_argument("--ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    parser.add_argument("--split", type=Split, choices=list(Split), default=Split.TEST)
    parser.add_argument("--max_clips", type=int, default=64)
    parser.add_argument("--window", type=int)
    parser.add_argument("--chunk_tokens", type=int, default=4)
    parser.add_argument("--max_lookahead", type=int, default=6)
    parser.add_argument("--chunk_grid", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--fps", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
