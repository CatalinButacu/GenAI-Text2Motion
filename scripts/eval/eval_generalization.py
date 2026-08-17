from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from text2motion.app.config import load_config
from text2motion.app.runtime import seed_everything
from text2motion.evaluation.matcher import load_eval_stats, load_matchers
from text2motion.tokenization.evaluation import evaluate_reconstruction
from text2motion.tokenization.model import build_tokenizer_module

CKPT_CONFIGS: dict[str, tuple[str, str]] = {
    "rvq_l4_512": ("configs/tokenizer/rvq_l4_512.yaml", "rvq"),
    "rvq_l6_512": ("configs/tokenizer/rvq_l6_512.yaml", "rvq"),
    "rvq_l8_512": ("configs/tokenizer/rvq_l8_512.yaml", "rvq"),
    "rvq_l4_1024": ("configs/tokenizer/rvq_l4_1024.yaml", "rvq"),
    "rvq_l6_1024": ("configs/tokenizer/rvq_l6_1024.yaml", "rvq"),
    "rvq_l8_1024": ("configs/tokenizer/rvq_l8_1024.yaml", "rvq"),
    "fsq_g4_v512": ("configs/tokenizer/fsq_g4_v512.yaml", "fsq"),
    "fsq_g6_v512": ("configs/tokenizer/tokenizer_isovocab.yaml", "fsq"),
    "fsq_g8_v512": ("configs/tokenizer/fsq_g8_v512.yaml", "fsq"),
    "fsq_g4_v1024": ("configs/tokenizer/fsq_g4_v1024.yaml", "fsq"),
    "fsq_g6_v1024": ("configs/tokenizer/fsq_g6_v1024.yaml", "fsq"),
    "fsq_g8_v1024": ("configs/tokenizer/fsq_g8_v1024.yaml", "fsq"),
    "fsq_g4_v1000": ("configs/tokenizer/tok_g4_v1000.yaml", "fsq"),
    "fsq_g6_v1000": ("configs/tokenizer/tok_g6_v1000.yaml", "fsq"),
    "fsq_g8_v1000": ("configs/tokenizer/tok_g8_v1000.yaml", "fsq"),
}
SPLITS = ("train", "val", "test")


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = load_config(next(iter(CKPT_CONFIGS.values()))[0])
    out_dir = Path(base.paths.hml3d_out_dir)
    our_mean = np.load(out_dir / "Mean.npy").astype(np.float32)
    our_std = np.load(out_dir / "Std.npy").astype(np.float32)
    eval_mean, eval_std = load_eval_stats(base.paths.eval_stats_dir)
    matcher, _ = load_matchers(base.paths.eval_matcher, device=device)
    joints = base.hml3d.num_joints
    ckpt_dir = Path(base.paths.checkpoints_dir)

    stems = [s for s in (args.only.split(",") if args.only else CKPT_CONFIGS) if s]
    rows: list[tuple[str, float, float, float]] = []
    print(
        f"recon-FID at {args.clips} seed-2026-shuffled clips/split (equal N -> gaps comparable)\n"
    )
    print(f"{'checkpoint':14} {'train':>8} {'val':>8} {'test':>8} {'gap(t-tr)':>10} {'|t-v|':>8}")
    for stem in stems:
        path = ckpt_dir / f"{stem}.pt"
        if not path.is_file():
            print(f"{stem:14} (no checkpoint, skipped)")
            continue
        cfg_path, mech = CKPT_CONFIGS[stem]
        cfg = load_config(cfg_path)
        seed_everything(cfg.seed, cfg.deterministic)
        tok, _ = build_tokenizer_module(mech, cfg)
        tok.load_state_dict(torch.load(path, map_location=device))
        tok.to(device).eval()
        fids = {
            sp: evaluate_reconstruction(
                tok,
                out_dir,
                our_mean,
                our_std,
                matcher,
                eval_mean,
                eval_std,
                joints_num=joints,
                device=device,
                max_clips=args.clips,
                split=sp,
                shuffle_seed=2026,
            )["recon_fid"]
            for sp in SPLITS
        }
        gap = fids["test"] - fids["train"]
        tv = abs(fids["test"] - fids["val"])
        rows.append((stem, fids["train"], fids["val"], fids["test"]))
        print(
            f"{stem:14} {fids['train']:8.4f} {fids['val']:8.4f} {fids['test']:8.4f} "
            f"{gap:+10.4f} {tv:8.4f}"
        )

    md = Path(base.paths.outputs_dir) / "generalization.md"
    lines = [
        f"# Tokenizer generalization audit (recon-FID, {args.clips} clips/split, seed 2026)",
        "",
        "Equal sample size across splits (FID is N-biased); the **gap = test - train** is the overfit",
        "indicator, **|test - val|** flags any test-selection inflation. Absolute values differ from the",
        "full-test headline by design.",
        "",
        "| checkpoint | train | val | test | gap (test-train) | \\|test-val\\| |",
        "|---|---|---|---|---|---|",
    ]
    for stem, tr, va, te in rows:
        lines.append(
            f"| {stem} | {tr:.4f} | {va:.4f} | {te:.4f} | {te - tr:+.4f} | {abs(te - va):.4f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {md}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tokenizer generalization gap (train/val/test recon-FID)."
    )
    p.add_argument(
        "--clips", type=int, default=1000, help="equal clip count per split (FID is N-biased)"
    )
    p.add_argument("--only", default="", help="comma-separated checkpoint stems (default: all)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
