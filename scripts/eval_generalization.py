"""Generalization audit for the trained tokenizers (Contribution A) -- NO re-training.

For each saved best checkpoint, reconstruct an EQUAL-SIZE, seed-shuffled sample of the
train / val / test splits and report recon-FID per split. Two diagnostics:
  * gap = test - train  : the overfitting indicator (large positive => memorised the train set);
  * |test - val|        : whether the historical test-based checkpoint selection inflated the headline
                          (val and test are both held out -> should be close).
FID is sample-size biased, so every split uses the SAME clip count (--clips); absolute values
therefore differ from the full-test headline -- it is the GAP that is meaningful.

Run when the GPU is free (this collides with an in-flight training run on the 4 GB card):
    $env:PYTHONPATH="src"; .venv/Scripts/python.exe scripts/eval_generalization.py --clips 1000
    $env:PYTHONPATH="src"; .venv/Scripts/python.exe scripts/eval_generalization.py --only fsq_g8_v512,rvq_l8_512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from text2motion.eval.matcher import load_eval_stats, load_matchers
from text2motion.eval.tokenizer_eval import evaluate_tokenizer
from text2motion.shared.config import load_config
from text2motion.shared.seed import seed_everything
from text2motion.train.train_tokenizer import build_tokenizer

# checkpoint stem -> (config it was trained with, mechanism). Grounded in the sweep scripts.
CKPT_CONFIGS: dict[str, tuple[str, str]] = {
    "rvq_l4_512": ("configs/tokenizer/rvq_l4_512.yaml", "rvq"),
    "rvq_l6_512": ("configs/tokenizer/rvq_l6_512.yaml", "rvq"),
    "rvq_l8_512": ("configs/tokenizer/rvq_l8_512.yaml", "rvq"),
    "rvq_l4_1024": ("configs/tokenizer/rvq_l4_1024.yaml", "rvq"),
    "rvq_l6_1024": ("configs/tokenizer/rvq_l6_1024.yaml", "rvq"),
    "rvq_l8_1024": ("configs/tokenizer/rvq_l8_1024.yaml", "rvq"),
    "fsq_g4_v512": ("configs/tokenizer/fsq_g4_v512.yaml", "fsq"),
    "fsq_g6_v512": ("configs/tokenizer/tokenizer_isovocab.yaml", "fsq"),  # 6x512 = (8,8,8)
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
    # The dataset / matcher / stats are identical across checkpoints -- load once.
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
    print(f"recon-FID at {args.clips} seed-2026-shuffled clips/split (equal N -> gaps comparable)\n")
    print(f"{'checkpoint':14} {'train':>8} {'val':>8} {'test':>8} {'gap(t-tr)':>10} {'|t-v|':>8}")
    for stem in stems:
        path = ckpt_dir / f"{stem}.pt"
        if not path.is_file():
            print(f"{stem:14} (no checkpoint, skipped)")
            continue
        cfg_path, mech = CKPT_CONFIGS[stem]
        cfg = load_config(cfg_path)
        seed_everything(cfg.seed, cfg.deterministic)
        tok, _ = build_tokenizer(mech, cfg)
        tok.load_state_dict(torch.load(path, map_location=device))
        tok.to(device).eval()
        fids = {
            sp: evaluate_tokenizer(
                tok, out_dir, our_mean, our_std, matcher, eval_mean, eval_std,
                joints_num=joints, device=device, max_clips=args.clips,
                split=sp, shuffle_seed=2026,
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
        lines.append(f"| {stem} | {tr:.4f} | {va:.4f} | {te:.4f} | {te - tr:+.4f} | {abs(te - va):.4f} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {md}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenizer generalization gap (train/val/test recon-FID).")
    p.add_argument("--clips", type=int, default=1000, help="equal clip count per split (FID is N-biased)")
    p.add_argument("--only", default="", help="comma-separated checkpoint stems (default: all)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
