from __future__ import annotations

import argparse
from pathlib import Path

import torch

from text2motion.data.hml3d.motion_window import build_window_loader
from text2motion.data.hml3d.stats import MotionScaler
from text2motion.eval.matcher import load_eval_stats, load_matchers
from text2motion.eval.tokenizer_eval import evaluate_tokenizer
from text2motion.model.rvq_baseline import RvqBaselineTokenizer
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import Config, load_config
from text2motion.shared.run_log import log_metrics, start_run
from text2motion.shared.seed import seed_everything
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
    seed_everything(cfg.seed, cfg.deterministic)  # reproducible + bit-exact (added 2026-06-15)

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
    scaler = MotionScaler.load(out_dir, dim=cfg.hml3d.dim)
    our_mean, our_std = scaler.mean, scaler.std
    eval_mean, eval_std = load_eval_stats(cfg.paths.eval_stats_dir)
    matcher, _ = load_matchers(cfg.paths.eval_matcher, device=device)

    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = args.ckpt_name or f"tokenizer_{args.tokenizer}.pt"
    ckpt_path = ckpt_dir / ckpt_name
    resume_path = ckpt_dir / f"{Path(ckpt_name).stem}_last.pt"
    run_dir = start_run(f"tokenizer_{Path(ckpt_name).stem}", cfg, cfg.paths.outputs_dir, vars(args))
    print(
        f"train clips: {len(loader.dataset)}  device: {device}  codebook: {tokenizer.codebook_size}"
    )

    best_fid = float("inf")
    start_epoch = 0
    if args.resume and resume_path.is_file():
        state = torch.load(resume_path, map_location=device)
        tokenizer.load_state_dict(state["tokenizer"])
        trainer.opt.load_state_dict(state["optimizer"])
        trainer.ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        start_epoch = state["epoch"] + 1
        best_fid = state["best_fid"]
        print(f"resumed from {resume_path} at epoch {start_epoch} (best recon-FID {best_fid:.4f})")

    for epoch in range(start_epoch, args.epochs):
        loader.dataset.set_epoch(epoch)
        tokenizer.train()
        totals: dict[str, float] = {}
        steps = 0
        for motion in loader:
            parts = trainer.train_step(motion.to(device))
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        means = {k: v / steps for k, v in totals.items()}
        log_metrics(run_dir, {"epoch": epoch + 1, **means})
        print(
            f"epoch {epoch + 1:3d}  [common] recon {means['recon']:.4f} total {means['total']:.4f}  "
            f"[health] perplexity {means['perplexity']:.1f} usage {means['usage_frac']:.1%} "
            f"commit {means['commit']:.4f}"
        )

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            trainer.ema.copy_to(tokenizer)
            metrics = evaluate_tokenizer(
                tokenizer,
                out_dir,
                our_mean,
                our_std,
                matcher,
                eval_mean,
                eval_std,
                joints_num=cfg.hml3d.num_joints,
                device=device,
                max_clips=args.max_eval_clips,
                split="val",  # selection on VAL only; TEST is evaluated once after the loop
            )
            trainer.ema.restore(tokenizer)
            log_metrics(run_dir, {"epoch": epoch + 1, "split": "val", **metrics})
            print(
                f"  [val] clips {metrics['clips']}  MPJPE {metrics['mpjpe_mm']:.1f}mm  "
                f"feat-L2 {metrics['feature_l2']:.4f}  recon-FID {metrics['recon_fid']:.4f}"
            )
            if metrics["recon_fid"] < best_fid:
                best_fid = metrics["recon_fid"]
                trainer.ema.copy_to(tokenizer)
                torch.save(tokenizer.state_dict(), ckpt_path)
                trainer.ema.restore(tokenizer)
                print(f"  saved best (val) -> {ckpt_path} (val recon-FID {best_fid:.4f})")
            torch.save(  # full state at every eval: an interrupt costs <= eval_every epochs
                {
                    "tokenizer": tokenizer.state_dict(),
                    "optimizer": trainer.opt.state_dict(),
                    "ema": trainer.ema.shadow,
                    "epoch": epoch,
                    "best_fid": best_fid,
                },
                resume_path,
            )

    if ckpt_path.is_file():
        tokenizer.load_state_dict(torch.load(ckpt_path, map_location=device))
        tokenizer.eval()
        test_metrics = evaluate_tokenizer(
            tokenizer,
            out_dir,
            our_mean,
            our_std,
            matcher,
            eval_mean,
            eval_std,
            joints_num=cfg.hml3d.num_joints,
            device=device,
            max_clips=args.max_eval_clips,
            split="test",
        )
        log_metrics(run_dir, {"epoch": args.epochs, "split": "test", "final": True, **test_metrics})
        print(
            f"  [TEST once | val-selected best] clips {test_metrics['clips']}  "
            f"recon-FID {test_metrics['recon_fid']:.4f}  MPJPE {test_metrics['mpjpe_mm']:.1f}mm"
        )


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
    parser.add_argument(
        "--ckpt_name", default=None, help="checkpoint filename (ablations must not clobber winners)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume from <ckpt_name stem>_last.pt"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
