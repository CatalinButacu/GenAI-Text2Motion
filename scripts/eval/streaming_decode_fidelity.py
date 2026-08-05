from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from text2motion.shared.config import load_config
from text2motion.shared.run_log import log_metrics, start_run
from text2motion.stream.decode import StreamingMotionDecoder, measure_decoder_context
from text2motion.train.train_tokenizer import build_tokenizer


def load_tokenizer(cfg, ckpt_path: str, kind: str):
    tokenizer, _ = build_tokenizer(kind, cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "tokenizer"):
            if key in state:
                state = state[key]
                break
    tokenizer.load_state_dict(state)
    return tokenizer.eval()


def load_clips(out_dir: str, split: str, max_clips: int, window: int) -> np.ndarray:
    mean = np.load(os.path.join(out_dir, "Mean.npy"))
    std = np.load(os.path.join(out_dir, "Std.npy"))
    ids = [line.strip() for line in open(os.path.join(out_dir, f"{split}.txt")) if line.strip()]
    clips = []
    for clip_id in ids:
        base = clip_id[1:] if clip_id.startswith("M") else clip_id
        path = os.path.join(out_dir, "new_joint_vecs", f"{base}.npy")
        if not os.path.exists(path):
            continue
        feats = np.load(path)
        if feats.shape[0] < window:
            continue
        clips.append((feats[:window] - mean) / std)
        if len(clips) >= max_clips:
            break
    if not clips:
        raise RuntimeError(f"no clips of at least {window} frames found in {out_dir}/{split}")
    return np.stack(clips), mean, std


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg, args.ckpt, args.kind)
    out_dir = cfg.paths.hml3d_out_dir
    clips, mean, std = load_clips(out_dir, args.split, args.max_clips, args.window)
    motion = torch.from_numpy(clips).float()
    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()

    measured_left, measured_right = measure_decoder_context(tokenizer)
    print(f"measured decoder context: left {measured_left} tokens, right {measured_right} tokens")

    run_dir = start_run("streaming_decode_fidelity", cfg, cfg.paths.outputs_dir)

    with torch.no_grad():
        indices = tokenizer.encode(motion)
        reference = tokenizer.decode(indices) * std_t + mean_t
    scale = reference.abs().mean()
    token_list = [indices[:, step] for step in range(indices.shape[1])]

    rows = []
    for lookahead in range(args.max_lookahead + 1):
        decoder = StreamingMotionDecoder(
            tokenizer,
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
            "latency_seconds": lookahead * cfg.tokenizer.downsample / args.fps,
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
            tokenizer, mean_t, std_t, chunk_tokens=chunk, left_context=0, lookahead=0
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
            "split": args.split,
            "clips": int(motion.shape[0]),
            "window_frames": args.window,
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
    parser.add_argument("--kind", default="fsq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_clips", type=int, default=64)
    parser.add_argument("--window", type=int, default=196)
    parser.add_argument("--chunk_tokens", type=int, default=4)
    parser.add_argument("--max_lookahead", type=int, default=6)
    parser.add_argument("--chunk_grid", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--fps", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
