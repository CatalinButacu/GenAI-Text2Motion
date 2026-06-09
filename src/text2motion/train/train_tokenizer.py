"""Train the motion tokenizer (Contribution A) and evaluate reconstruction + downstream FID.

Trains either the Residual-FSQ tokenizer or the strong-RVQ baseline on fixed-length motion windows,
logging reconstruction loss + codebook perplexity per epoch and MPJPE / downstream FID on the test
split periodically (with the EMA weights). The best checkpoint (lowest downstream FID) is saved. Run
both and compare the recorded numbers -- that table is Contribution A's evidence.

    python -m text2motion.train.train_tokenizer --config configs/default.yaml --tokenizer fsq --epochs 50
    python -m text2motion.train.train_tokenizer --config configs/default.yaml --tokenizer rvq --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.motion_window import build_window_loader
from text2motion.eval.matcher import load_eval_stats, load_matchers
from text2motion.eval.tokenizer_eval import evaluate_tokenizer
from text2motion.model.rvq_baseline import RvqBaselineTokenizer
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import Config, load_config
from text2motion.train.tokenizer_trainer import TokenizerTrainer


def build_tokenizer(name: str, cfg: Config) -> tuple[torch.nn.Module, float]:
    if name == "fsq":
        return ResidualFsqTokenizer(cfg.tokenizer), 0.0
    if name == "rvq":
        return RvqBaselineTokenizer(cfg.rvq_baseline), cfg.rvq_baseline.commitment_beta
    raise ValueError(f"--tokenizer must be 'fsq' or 'rvq', got {name!r}")


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    loader = build_window_loader(
        cfg.paths, cfg.hml3d, cfg.data, "train", args.window, args.batch_size, args.num_workers
    )
    tokenizer, commit_beta = build_tokenizer(args.tokenizer, cfg)
    tokenizer.to(device)
    trainer = TokenizerTrainer(
        tokenizer,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        ema_decay=args.ema_decay,
        commit_beta=commit_beta,
    )

    out_dir = Path(cfg.paths.hml3d_out_dir)
    our_mean = np.load(out_dir / "Mean.npy").astype(np.float32)
    our_std = np.load(out_dir / "Std.npy").astype(np.float32)
    eval_mean, eval_std = load_eval_stats(cfg.paths.eval_stats_dir)
    matcher, _ = load_matchers(cfg.paths.eval_matcher, device=device)

    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"tokenizer_{args.tokenizer}.pt"
    print(f"train clips: {len(loader.dataset)}  device: {device}  codebook: {tokenizer.codebook_size}")

    best_fid = float("inf")
    for epoch in range(args.epochs):
        tokenizer.train()
        totals: dict[str, float] = {}
        steps = 0
        for motion in loader:
            parts = trainer.train_step(motion.to(device))
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        means = {k: v / steps for k, v in totals.items()}
        print(
            f"epoch {epoch + 1:3d}  recon {means['recon']:.4f}  commit {means['commit']:.4f}  "
            f"perplexity {means['perplexity']:.1f}"
        )

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            trainer.ema.copy_to(tokenizer)
            metrics = evaluate_tokenizer(
                tokenizer, out_dir, our_mean, our_std, matcher, eval_mean, eval_std,
                joints_num=cfg.hml3d.num_joints, device=device, max_clips=args.max_eval_clips,
            )
            trainer.ema.restore(tokenizer)
            print(
                f"  [eval] clips {metrics['clips']}  MPJPE {metrics['mpjpe_mm']:.1f}mm  "
                f"feat-L2 {metrics['feature_l2']:.4f}  recon-FID {metrics['recon_fid']:.4f}"
            )
            if metrics["recon_fid"] < best_fid:
                best_fid = metrics["recon_fid"]
                trainer.ema.copy_to(tokenizer)
                torch.save(tokenizer.state_dict(), ckpt_path)
                trainer.ema.restore(tokenizer)
                print(f"  saved best -> {ckpt_path} (recon-FID {best_fid:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the motion tokenizer (Contribution A).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True, choices=["fsq", "rvq"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--max_eval_clips", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
