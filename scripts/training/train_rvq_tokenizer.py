"""Train the RVQ tokenizer on raw SMPL-X pose sequences.

This is a pre-training step for TextToMotionSSM. The tokenizer learns to:
    encode (T, 168) pose -> (T/4, K) codebook indices
    decode (T/4, K) codebook indices -> (T, 168) pose

Run this first, then `train_motion_ssm.py` which freezes the tokenizer and
trains the SSM transformer to predict codebook indices from text.

Usage:
    python scripts/training/train_rvq_tokenizer.py --data-dir data/AMASS
    python scripts/training/train_rvq_tokenizer.py --epochs 30 --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.architecture.rvq_tokenizer import MotionRVQTokenizer
from src.data.motion_dataset import MotionDataset
from src.data.motion_normalize import MotionStats
from src.data.unified import build_or_load_unified_buffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig, UnifiedMotionDataset
from src.shared.constants import CONSTS
from src.shared.run_ctx import (
    init_wandb,
    log_gpu_sanity,
    make_run_dir,
    snapshot_config,
    wandb_finish,
    wandb_log,
)

log = logging.getLogger(__name__)


def compute_tokenizer_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    commit_loss: torch.Tensor,
    recon_weight: float = 1.0,
    vel_weight: float = 0.5,
    commit_weight: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    """Compose the RVQ tokenizer training loss.

    recon_loss  -- MSE reconstruction fidelity (MoMask weight: 1.0)
    vel_loss    -- MSE on frame-to-frame differences, penalises jitter (0.5)
    commit_loss -- averaged commitment loss from ResidualVectorQuantizer (0.02)

    Lower commit_weight risks codebook collapse; higher blocks encoder exploration.
    Higher vel_weight smooths motion but washes out sharp transitions (kicks, jumps).
    """
    recon_loss = F.mse_loss(recon, target)
    vel_recon = recon[:, 1:] - recon[:, :-1]
    vel_target = target[:, 1:] - target[:, :-1]
    vel_loss = F.mse_loss(vel_recon, vel_target)
    total = recon_weight * recon_loss + vel_weight * vel_loss + commit_weight * commit_loss
    return total, {
        "loss": total.detach(),
        "recon": recon_loss.detach(),
        "vel": vel_loss.detach(),
        "commit": commit_loss.detach(),
    }


def train_epoch(
    model: MotionRVQTokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    recon_weight: float,
    vel_weight: float,
    commit_weight: float,
) -> dict:
    model.train()
    total: dict = {}
    n = 0

    for batch in loader:
        motion = batch["motion"].to(device)  # (B, T, 168)

        recon, _, commit_loss = model(motion)
        loss, parts = compute_tokenizer_loss(
            recon, motion, commit_loss, recon_weight, vel_weight, commit_weight
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in parts.items():
            total[k] = total.get(k, 0.0) + float(v)
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def validate(
    model: MotionRVQTokenizer, loader: DataLoader, device: torch.device
) -> tuple[float, list[dict]]:
    model.eval()
    total_recon = 0.0
    n = 0

    for batch in loader:
        motion = batch["motion"].to(device)
        recon, _, _ = model(motion)
        total_recon += float(F.mse_loss(recon, motion))
        n += 1

    util = model.codebook_utilization()
    return total_recon / max(n, 1), util


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/AMASS", dest="data_dir")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128, dest="latent_dim")
    parser.add_argument("--n-codebooks", type=int, default=6, dest="n_codebooks")
    parser.add_argument("--codebook-size", type=int, default=512, dest="codebook_size")
    parser.add_argument("--down-t", type=int, default=4, dest="down_t")
    parser.add_argument("--max-motion-length", type=int, default=200, dest="max_motion_length")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/rvq_tokenizer", dest="checkpoint_dir"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0=main process, safe on Windows)",
        dest="num_workers",
    )
    # Loss weights (MoMask defaults -- tune per ablation)
    parser.add_argument(
        "--recon-weight",
        type=float,
        default=1.0,
        help="Reconstruction MSE weight (MoMask=1.0)",
        dest="recon_weight",
    )
    parser.add_argument(
        "--vel-weight",
        type=float,
        default=0.5,
        help="Velocity smoothness MSE weight (MoMask=0.5, Mogo=1.0)",
        dest="vel_weight",
    )
    parser.add_argument(
        "--commit-weight",
        type=float,
        default=0.25,
        help="Codebook commitment loss weight (MoMask=0.02, Mogo=0.25)",
        dest="commit_weight",
    )
    parser.add_argument(
        "--reset-dead-every",
        type=int,
        default=5,
        dest="reset_dead_every",
        help=(
            "Run VQ-VAE-2 dead-code revival every N epochs (0 disables). "
            "Replaces unused codebook entries with random training latents "
            "to fight collapse on deeper RVQ layers."
        ),
    )
    parser.add_argument(
        "--reset-dead-threshold",
        type=float,
        default=1.0,
        dest="reset_dead_threshold",
        help="Cluster_size threshold below which a code is considered dead.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        dest="weight_decay",
        help="AdamW weight decay (L2 reg).",
    )
    # Resume / data source / wandb
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=("Path to checkpoint .pt to resume training from (model+optimizer+scheduler+epoch)"),
    )
    parser.add_argument(
        "--causal-decoder",
        action="store_true",
        dest="causal_decoder",
        help=(
            "Build the decoder with CausalConv1d + nearest-neighbor "
            "upsampling instead of ConvTranspose1d. Required for "
            "streaming-mode inference (output frame t depends only "
            "on latent tokens up to ceil(t / down_t)). Default off "
            "preserves bit-compatibility with existing checkpoints."
        ),
    )
    parser.add_argument(
        "--data",
        default="amass",
        choices=["amass", "humanml3d", "all", "mega"],
        dest="data_source",
        help=(
            "Which corpus to train on: amass (default), humanml3d, "
            "all (AMASS+HumanML3D), or mega (AMASS+HumanML3D+ARCTIC+InterX)."
        ),
    )
    parser.add_argument(
        "--humanml3d-dir",
        default="data/humanml3d",
        dest="humanml3d_dir",
        help="HumanML3D root (used when --data is humanml3d or all/mega)",
    )
    parser.add_argument(
        "--arctic-dir",
        default="data/arctic/unpack",
        dest="arctic_dir",
        help="ARCTIC root (used when --data is mega)",
    )
    parser.add_argument(
        "--interx-dir",
        default="data/inter-x",
        dest="interx_dir",
        help="Inter-X root (used when --data is mega)",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        dest="stats_path",
        help=(
            "Path to a precomputed (mean, std) .npz from "
            "scripts/training/precompute_stats.py. Required for fair "
            "cross-corpus comparison; without it, each --data run uses "
            "its own train-split stats and val_recon is incomparable."
        ),
    )
    parser.add_argument(
        "--trans-stats-path",
        type=str,
        default=None,
        dest="trans_stats_path",
        help=(
            "Path to per-source translation stats .npz from "
            "scripts/training/precompute_translation_stats.py. When set, "
            "translation channels (3:6) are normalised per source "
            "(amass / humanml3d) instead of via the shared stats — "
            "fixes the bimodal AMASS-vs-HumanML3D translation distribution."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        default="offline",
        choices=["online", "offline", "disabled"],
        dest="wandb_mode",
        help=("W&B mode: offline (default, sync later), online (needs WANDB_API_KEY), disabled"),
    )
    args = parser.parse_args()

    run_dir, run_id = make_run_dir(args.checkpoint_dir)
    args.checkpoint_dir = str(run_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "training.log", mode="a"),
        ],
    )
    log.info("[train_rvq] run_id=%s  run_dir=%s", run_id, run_dir)
    log_gpu_sanity()
    snapshot_config(run_dir, vars(args))

    os.environ["WANDB_MODE"] = args.wandb_mode
    init_wandb(project="rvq_tokenizer", run_id=run_id, run_dir=run_dir, config=vars(args))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    shared_stats = None
    if args.stats_path:
        loaded = np.load(args.stats_path)
        shared_stats = MotionStats(mean=loaded["mean"], std=loaded["std"])
        log.info(
            "[rvq] using shared stats from %s  mean|.|=%.4f  std|.|=%.4f",
            args.stats_path,
            float(np.abs(shared_stats.mean).mean()),
            float(np.abs(shared_stats.std).mean()),
        )
    else:
        log.warning(
            "[rvq] no --stats-path: this run computes its own stats. "
            "val_recon will NOT be comparable across --data variants."
        )

    trans_stats_by_source: dict[str, MotionStats] | None = None
    if args.trans_stats_path:
        ts = np.load(args.trans_stats_path)
        trans_stats_by_source = {}

        for src_key in ("amass", "humanml3d", "arctic", "interx"):
            mean_k = f"{src_key}_mean"
            std_k = f"{src_key}_std"

            if mean_k in ts.files and std_k in ts.files:
                trans_stats_by_source[src_key] = MotionStats(mean=ts[mean_k], std=ts[std_k])
        log.info(
            "[rvq] per-source translation stats from %s (sources=%s)",
            args.trans_stats_path,
            list(trans_stats_by_source.keys()),
        )

        for src_key, st in trans_stats_by_source.items():
            log.info("[rvq]   %s trans std|.|=%.4f", src_key, float(np.abs(st.std).mean()))

    if args.data_source == "amass":
        log.info("[rvq] data source: AMASS only (%s)", args.data_dir)
        train_ds = MotionDataset(
            args.data_dir,
            "train",
            args.max_motion_length,
            augment=True,
            stats=shared_stats,
            trans_stats_by_source=trans_stats_by_source,
        )
        val_ds = MotionDataset(
            args.data_dir,
            "val",
            args.max_motion_length,
            vocab=train_ds.vocab,
            stats=train_ds.motion_stats,
            trans_stats_by_source=trans_stats_by_source,
        )
        test_ds = MotionDataset(
            args.data_dir,
            "test",
            args.max_motion_length,
            vocab=train_ds.vocab,
            stats=train_ds.motion_stats,
            trans_stats_by_source=trans_stats_by_source,
        )
    else:
        if args.data_source == "humanml3d":
            log.info(
                "[rvq] data source: HumanML3D only (%s, AMASS backing=%s)",
                args.humanml3d_dir,
                args.data_dir,
            )
            cfg = UnifiedConfig(
                amass=SourceConfig(enabled=False),
                arctic=SourceConfig(enabled=False),
                humanml3d=SourceConfig(
                    enabled=True, data_dir=args.humanml3d_dir, amass_dir=args.data_dir
                ),
            )
        elif args.data_source == "all":
            log.info("[rvq] data source: AMASS + HumanML3D unified")
            cfg = UnifiedConfig(
                amass=SourceConfig(enabled=True, data_dir=args.data_dir),
                arctic=SourceConfig(enabled=False),
                humanml3d=SourceConfig(
                    enabled=True, data_dir=args.humanml3d_dir, amass_dir=args.data_dir
                ),
            )
        else:  # "mega"
            log.info("[rvq] data source: AMASS + HumanML3D + ARCTIC + InterX unified")
            cfg = UnifiedConfig(
                amass=SourceConfig(enabled=True, data_dir=args.data_dir),
                arctic=SourceConfig(enabled=True, data_dir=args.arctic_dir),
                humanml3d=SourceConfig(
                    enabled=True, data_dir=args.humanml3d_dir, amass_dir=args.data_dir
                ),
                interx=SourceConfig(enabled=True, data_dir=args.interx_dir),
            )
        buf = build_or_load_unified_buffer(cfg, args.data_source)
        log.info("[rvq] buffer: %d samples", len(buf))
        train_ds = UnifiedMotionDataset(
            "train",
            args.max_motion_length,
            augment=True,
            preloaded_buf=buf,
            stats=shared_stats,
            trans_stats_by_source=trans_stats_by_source,
        )
        val_ds = UnifiedMotionDataset(
            "val",
            args.max_motion_length,
            preloaded_buf=buf,
            vocab=train_ds.vocab,
            stats=train_ds.motion_stats,
            trans_stats_by_source=trans_stats_by_source,
        )
        test_ds = UnifiedMotionDataset(
            "test",
            args.max_motion_length,
            preloaded_buf=buf,
            vocab=train_ds.vocab,
            stats=train_ds.motion_stats,
            trans_stats_by_source=trans_stats_by_source,
        )
    log.info("[rvq] dataset: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = MotionRVQTokenizer(
        motion_dim=CONSTS.smplx.pose_dim,
        latent_dim=args.latent_dim,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        down_t=args.down_t,
        causal_decoder=args.causal_decoder,
    ).to(device)
    log.info("[rvq] params: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    start_epoch = 1

    if args.resume:
        log.info("[rvq] resuming from %s", args.resume)
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = int(ck["epoch"]) + 1
        best_val = float(ck.get("val_loss", float("inf")))
        log.info("[rvq] resumed at epoch=%d  best_val=%.4f", start_epoch, best_val)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.recon_weight,
            args.vel_weight,
            args.commit_weight,
        )

        if args.reset_dead_every > 0 and epoch % args.reset_dead_every == 0:
            sample_batch = next(iter(train_loader))
            reset_counts = model.reset_dead_codes(
                sample_batch["motion"].to(device),
                threshold=args.reset_dead_threshold,
            )

            if sum(reset_counts) > 0:
                log.info("[rvq] dead-code reset @ epoch %d: %s", epoch, reset_counts)
        val_recon, util = validate(model, val_loader, device)
        scheduler.step()

        mean_active = sum(u["active_fraction"] for u in util) / max(len(util), 1)
        mean_entropy_pct = sum(u["entropy"] / max(u["max_entropy"], 1e-8) for u in util) / max(
            len(util), 1
        )
        log.info(
            "[rvq] epoch %2d  loss=%.4f  recon=%.4f  vel=%.4f  val_recon=%.4f  "
            "cb_active=%.1f%%  cb_entropy=%.1f%%  (%.1fs)",
            epoch,
            train_metrics.get("loss", 0.0),
            train_metrics.get("recon", 0.0),
            train_metrics.get("vel", 0.0),
            val_recon,
            mean_active * 100,
            mean_entropy_pct * 100,
            time.time() - t0,
        )
        ck = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_recon,
            "config": vars(args),
        }
        torch.save(ck, os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}.pt"))

        if val_recon < best_val:
            best_val = val_recon
            torch.save(ck, os.path.join(args.checkpoint_dir, "best_model.pt"))
            log.info("[rvq] best model saved (val_recon=%.4f)", val_recon)

        # wandb_log runs AFTER on-disk saves so a wandb-side crash never costs an epoch
        util_log = {f"codebook/cb{i}_{k}": v for i, u in enumerate(util) for k, v in u.items()}
        wandb_log(
            {
                "epoch": epoch,
                "train/loss": train_metrics.get("loss", 0.0),
                "train/recon": train_metrics.get("recon", 0.0),
                "train/vel": train_metrics.get("vel", 0.0),
                "train/commit": train_metrics.get("commit", 0.0),
                "val/recon": val_recon,
                "best_val_recon": min(best_val, val_recon),
                "codebook/mean_active_pct": mean_active * 100,
                "codebook/mean_entropy_pct": mean_entropy_pct * 100,
                **util_log,
            },
            step=epoch,
        )

    log.info("[rvq] done. best val_recon=%.4f", best_val)

    # --- Final test evaluation on held-out 10% ---
    log.info("[rvq] loading best checkpoint for test evaluation...")
    best_ck = torch.load(
        os.path.join(args.checkpoint_dir, "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ck["model_state_dict"])
    test_recon, test_util = validate(model, test_loader, device)
    test_active = sum(u["active_fraction"] for u in test_util) / max(len(test_util), 1)
    log.info(
        "[rvq] TEST recon=%.4f  cb_active=%.1f%%",
        test_recon,
        test_active * 100,
    )
    wandb_log(
        {
            "test/recon": test_recon,
            "test/codebook_active_pct": test_active * 100,
        }
    )

    wandb_finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
