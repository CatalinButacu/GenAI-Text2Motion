from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from text2motion.shared.config import load_config
from text2motion.shared.run_log import log_metrics, start_run


def base_ids(split_file: Path) -> list[str]:
    raw = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()]
    present = [name for name in raw if name]
    stripped = [name[1:] if name.startswith("M") else name for name in present]
    return sorted(set(stripped))


def split_frame_counts(out_dir: Path, split: str, max_motion_len: int) -> tuple[int, int, int]:
    clip_count = 0
    frames_total = 0
    frames_windowed = 0
    for clip_id in base_ids(out_dir / f"{split}.txt"):
        feature_file = out_dir / "new_joint_vecs" / f"{clip_id}.npy"
        if not feature_file.exists():
            continue
        length = int(np.load(feature_file, mmap_mode="r").shape[0])
        clip_count += 1
        frames_total += length
        frames_windowed += min(length, max_motion_len)
    return clip_count, frames_total, frames_windowed


def tokens_from_frames(frames: int, downsample: int, num_quantizers: int) -> int:
    return frames // downsample * num_quantizers


def amass_corpus_frames(amass_dir: Path) -> tuple[int, int]:
    files = sorted(amass_dir.rglob("*.npy"))
    frames = sum(int(np.load(path, mmap_mode="r").shape[0]) for path in files)
    return len(files), frames


def amass_token_file_total(token_file: Path, num_quantizers: int) -> tuple[int, int]:
    bundle = np.load(token_file, allow_pickle=True)
    latent_steps = sum(int(bundle[key].shape[0]) for key in bundle.files)
    return len(bundle.files), latent_steps * num_quantizers


def effective_tokens(unique_tokens: int, epochs: int, repeat_half_life: float) -> float:
    decayed = repeat_half_life * (1.0 - math.exp(-(epochs - 1) / repeat_half_life))
    return unique_tokens * (1.0 + decayed)


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    out_dir = Path(cfg.paths.hml3d_out_dir)
    downsample = cfg.tokenizer.downsample
    num_quantizers = cfg.tokenizer.num_quantizers
    vocab_bits = math.log2(args.codebook_size)
    max_motion_len = cfg.data.max_motion_len

    run_dir = start_run("scaling_budget", cfg, Path(cfg.paths.outputs_dir))

    split_rows = {}
    for split in ("train", "val", "test"):
        clips, frames, windowed = split_frame_counts(out_dir, split, max_motion_len)
        tokens = tokens_from_frames(windowed, downsample, num_quantizers)
        split_rows[split] = {
            "clips": clips,
            "frames": frames,
            "frames_windowed": windowed,
            "hours": frames / args.fps / 3600.0,
            "tokens": tokens,
        }
        print(f"{split:6s} clips {clips:6d} frames {frames:9d} tokens {tokens:10d}")

    mirror_factor = 2 if cfg.data.mirror_augment else 1
    train_tokens = split_rows["train"]["tokens"] * mirror_factor
    print(f"train tokens with mirror x{mirror_factor}: {train_tokens}")

    amass_dir = Path(args.amass_dir)
    amass_clips, amass_frames = amass_corpus_frames(amass_dir)
    amass_all_tokens = tokens_from_frames(amass_frames, downsample, num_quantizers)
    amass_windows, amass_used_tokens = amass_token_file_total(
        Path(args.amass_tokens), num_quantizers
    )
    print(
        f"amass on disk: clips {amass_clips} frames {amass_frames} tokens_if_all {amass_all_tokens}"
    )
    print(f"amass tokenized: windows {amass_windows} tokens {amass_used_tokens}")

    entropy_bits = train_tokens * vocab_bits
    print(f"train-set maximum entropy: {entropy_bits / 1e6:.1f} Mbit = {entropy_bits / 8e6:.2f} MB")

    model_rows = {}
    for label, params in args.model_params:
        chinchilla_tokens = args.tokens_per_param * params
        capacity_bits = args.bits_per_param * params
        model_rows[label] = {
            "params": params,
            "chinchilla_tokens": chinchilla_tokens,
            "shortfall_factor": chinchilla_tokens / train_tokens,
            "tokens_per_param": train_tokens / params,
            "memorization_bits": capacity_bits,
            "capacity_over_entropy": capacity_bits / entropy_bits,
        }
        print(
            f"{label:14s} params {params / 1e6:6.1f}M "
            f"chinchilla_D {chinchilla_tokens / 1e9:5.2f}B "
            f"shortfall x{chinchilla_tokens / train_tokens:6.0f} "
            f"capacity/entropy {capacity_bits / entropy_bits:4.1f}x"
        )

    repetition_rows = {}
    for epochs in args.epoch_grid:
        effective = effective_tokens(train_tokens, epochs, args.repeat_half_life)
        repetition_rows[str(epochs)] = {
            "effective_tokens": effective,
            "optimal_params": effective / args.tokens_per_param,
        }
        print(
            f"epochs {epochs:4d} effective {effective / 1e6:7.1f}M "
            f"optimal_params {effective / args.tokens_per_param / 1e6:5.2f}M"
        )

    ceiling_tokens = train_tokens * (1.0 + args.repeat_half_life)
    tokens_per_hour = args.fps * 3600 / downsample * num_quantizers
    hours_needed = args.tokens_per_param * args.chinchilla_target_params / tokens_per_hour
    print(f"repetition ceiling {ceiling_tokens / 1e6:.1f}M tokens")
    print(
        f"hours of capture to feed {args.chinchilla_target_params / 1e6:.0f}M params: {hours_needed:.0f}"
    )

    log_metrics(
        run_dir,
        {
            "downsample": downsample,
            "num_quantizers": num_quantizers,
            "codebook_size": args.codebook_size,
            "splits": split_rows,
            "train_tokens_mirrored": train_tokens,
            "amass_clips": amass_clips,
            "amass_frames": amass_frames,
            "amass_tokens_if_all": amass_all_tokens,
            "amass_tokens_used": amass_used_tokens,
            "train_entropy_bits": entropy_bits,
            "models": model_rows,
            "repetition": repetition_rows,
            "repetition_ceiling_tokens": ceiling_tokens,
            "tokens_per_hour": tokens_per_hour,
            "hours_to_feed_target": hours_needed,
        },
    )
    print(f"run dir: {run_dir}")


def parse_model_params(value: str) -> tuple[str, float]:
    label, raw = value.split("=")
    return label, float(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/final100m_fsq8x1024.yaml")
    parser.add_argument("--amass_dir", default="data/AMASS_263")
    parser.add_argument("--amass_tokens", default="data/amass_tokens_fsq8x1024.npz")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--tokens_per_param", type=float, default=20.0)
    parser.add_argument("--bits_per_param", type=float, default=2.0)
    parser.add_argument("--repeat_half_life", type=float, default=15.4)
    parser.add_argument("--chinchilla_target_params", type=float, default=98.07e6)
    parser.add_argument("--epoch_grid", type=int, nargs="+", default=[18, 30, 60, 290])
    parser.add_argument(
        "--model_params",
        type=parse_model_params,
        nargs="+",
        default=[
            ("twin-34M", 33.91e6),
            ("transformer-100M", 98.07e6),
            ("T2M-GPT", 228.0e6),
        ],
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
