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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.augmentation import detectTpose, qualityFilter, resampleToFps
from src.data.dataset_cache import INGEST_MAX_LENGTH
from src.data.motion_dataset import MotionDataset
from src.data.motion_normalize import MotionStats
from src.data.unified import buildSourcesBuffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig, UnifiedMotionDataset
from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer
from src.shared.constants import MOTION_DIM
from src.shared.run_ctx import (
    initWandb,
    logGpuSanity,
    makeRunDir,
    snapshotConfig,
    wandbFinish,
    wandbLog,
)

log = logging.getLogger(__name__)


def computeTokenizerLoss(
    recon: torch.Tensor,
    target: torch.Tensor,
    commitLoss: torch.Tensor,
    reconWeight: float = 1.0,
    velWeight: float = 0.5,
    commitWeight: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    """Compose the RVQ tokenizer training loss.

    recon_loss  -- MSE reconstruction fidelity (MoMask weight: 1.0)
    vel_loss    -- MSE on frame-to-frame differences, penalises jitter (0.5)
    commitLoss -- averaged commitment loss from ResidualVectorQuantizer (0.02)

    Lower commitWeight risks codebook collapse; higher blocks encoder exploration.
    Higher velWeight smooths motion but washes out sharp transitions (kicks, jumps).
    """
    reconLoss = F.mse_loss(recon, target)
    velRecon = recon[:, 1:] - recon[:, :-1]
    velTarget = target[:, 1:] - target[:, :-1]
    velLoss = F.mse_loss(velRecon, velTarget)
    total = reconWeight * reconLoss + velWeight * velLoss + commitWeight * commitLoss
    return total, {
        "loss": total.detach(),
        "recon": reconLoss.detach(),
        "vel": velLoss.detach(),
        "commit": commitLoss.detach(),
    }


def trainEpoch(
    model: MotionRVQTokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reconWeight: float,
    velWeight: float,
    commitWeight: float,
) -> dict:
    model.train()
    total: dict = {}
    n = 0

    for batch in loader:
        motion = batch["motion"].to(device)  # (B, T, 168)

        recon, _, commit_loss = model(motion)
        loss, parts = computeTokenizerLoss(
            recon, motion, commit_loss, reconWeight, velWeight, commitWeight
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
    totalRecon = 0.0
    n = 0

    for batch in loader:
        motion = batch["motion"].to(device)
        recon, _, _ = model(motion)
        totalRecon += float(F.mse_loss(recon, motion))
        n += 1

    util = model.codebookUtilization()
    return totalRecon / max(n, 1), util


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/AMASS", dest="dataDir")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32, dest="batchSize")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128, dest="latentDim")
    parser.add_argument("--n-codebooks", type=int, default=6, dest="nCodebooks")
    parser.add_argument("--codebook-size", type=int, default=512, dest="codebookSize")
    parser.add_argument("--down-t", type=int, default=4, dest="downT")
    parser.add_argument("--max-motion-length", type=int, default=200, dest="maxMotionLength")
    parser.add_argument("--checkpoint-dir", default="checkpoints/rvq_tokenizer",
                        dest="checkpointDir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0=main process, safe on Windows)",
                        dest="numWorkers")
    # Loss weights (MoMask defaults -- tune per ablation)
    parser.add_argument("--recon-weight", type=float, default=1.0,
                        help="Reconstruction MSE weight (MoMask=1.0)", dest="reconWeight")
    parser.add_argument("--vel-weight", type=float, default=0.5,
                        help="Velocity smoothness MSE weight (MoMask=0.5, Mogo=1.0)",
                        dest="velWeight")
    parser.add_argument("--commit-weight", type=float, default=0.02,
                        help="Codebook commitment loss weight (MoMask=0.02, Mogo=0.25)",
                        dest="commitWeight")
    # Resume / data source / wandb
    parser.add_argument("--resume", type=str, default=None,
                        help=("Path to checkpoint .pt to resume training from "
                              "(model+optimizer+scheduler+epoch)"))
    parser.add_argument("--data", default="amass", choices=["amass", "humanml3d", "all"],
                        dest="dataSource",
                        help=("Which corpus to train on: amass (default), "
                              "humanml3d, or all (unified)"))
    parser.add_argument("--humanml3d-dir", default="data/humanml3d", dest="humanml3dDir",
                        help="HumanML3D root (used when --data is humanml3d or all)")
    parser.add_argument("--stats-path", type=str, default=None, dest="statsPath",
                        help=("Path to a precomputed (mean, std) .npz from "
                              "scripts/training/precompute_stats.py. Required for fair "
                              "cross-corpus comparison; without it, each --data run uses "
                              "its own train-split stats and val_recon is incomparable."))
    parser.add_argument("--wandb-mode", default="offline",
                        choices=["online", "offline", "disabled"],
                        dest="wandbMode",
                        help=("W&B mode: offline (default, sync later), "
                              "online (needs WANDB_API_KEY), disabled"))
    args = parser.parse_args()

    run_dir, run_id = makeRunDir(args.checkpointDir)
    args.checkpointDir = str(run_dir)
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
    logGpuSanity()
    snapshotConfig(run_dir, vars(args))

    os.environ["WANDB_MODE"] = args.wandbMode
    initWandb(project="rvq_tokenizer", runId=run_id, runDir=run_dir, config=vars(args))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    sharedStats = None
    if args.statsPath:
        loaded = np.load(args.statsPath)
        sharedStats = MotionStats(mean=loaded["mean"], std=loaded["std"])
        log.info("[rvq] using shared stats from %s  mean|.|=%.4f  std|.|=%.4f",
                 args.statsPath,
                 float(np.abs(sharedStats.mean).mean()),
                 float(np.abs(sharedStats.std).mean()))
    else:
        log.warning("[rvq] no --stats-path: this run computes its own stats. "
                    "val_recon will NOT be comparable across --data variants.")

    if args.dataSource == "amass":
        log.info("[rvq] data source: AMASS only (%s)", args.dataDir)
        trainDs = MotionDataset(
            args.dataDir, "train", args.maxMotionLength, augment=True,
            stats=sharedStats,
        )
        valDs = MotionDataset(
            args.dataDir, "val", args.maxMotionLength,
            vocab=trainDs.vocab, stats=trainDs.motion_stats,
        )
        testDs = MotionDataset(
            args.dataDir, "test", args.maxMotionLength,
            vocab=trainDs.vocab, stats=trainDs.motion_stats,
        )
    else:
        if args.dataSource == "humanml3d":
            log.info("[rvq] data source: HumanML3D only (%s, AMASS backing=%s)",
                     args.humanml3dDir, args.dataDir)
            cfg = UnifiedConfig(
                amass=SourceConfig(enabled=False),
                arctic=SourceConfig(enabled=False),
                humanml3d=SourceConfig(enabled=True, dataDir=args.humanml3dDir,
                                       amassDir=args.dataDir),
            )
        else:  # "all"
            log.info("[rvq] data source: AMASS + HumanML3D unified")
            cfg = UnifiedConfig(
                amass=SourceConfig(enabled=True, dataDir=args.dataDir),
                arctic=SourceConfig(enabled=False),
                humanml3d=SourceConfig(enabled=True, dataDir=args.humanml3dDir,
                                       amassDir=args.dataDir),
            )
        buf = buildSourcesBuffer(cfg, 30, resampleToFps, qualityFilter, detectTpose,
                                 maxLength=INGEST_MAX_LENGTH)
        log.info("[rvq] buffer: %d samples", len(buf))
        trainDs = UnifiedMotionDataset(
            "train", args.maxMotionLength, augment=True, preloadedBuf=buf,
            stats=sharedStats,
        )
        valDs = UnifiedMotionDataset(
            "val", args.maxMotionLength, preloadedBuf=buf,
            vocab=trainDs.vocab, stats=trainDs.motion_stats,
        )
        testDs = UnifiedMotionDataset(
            "test", args.maxMotionLength, preloadedBuf=buf,
            vocab=trainDs.vocab, stats=trainDs.motion_stats,
        )
    log.info("[rvq] dataset: train=%d val=%d test=%d", len(trainDs), len(valDs), len(testDs))

    trainLoader = DataLoader(
        trainDs, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorkers
    )
    valLoader = DataLoader(
        valDs, batch_size=args.batchSize, shuffle=False, num_workers=args.numWorkers
    )
    testLoader = DataLoader(
        testDs, batch_size=args.batchSize, shuffle=False, num_workers=args.numWorkers
    )

    model = MotionRVQTokenizer(
        motionDim=MOTION_DIM,
        latentDim=args.latentDim,
        nCodebooks=args.nCodebooks,
        codebookSize=args.codebookSize,
        downT=args.downT,
    ).to(device)
    log.info("[rvq] params: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    bestVal = float("inf")
    startEpoch = 1

    if args.resume:
        log.info("[rvq] resuming from %s", args.resume)
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        startEpoch = int(ck["epoch"]) + 1
        bestVal = float(ck.get("val_loss", float("inf")))
        log.info("[rvq] resumed at epoch=%d  best_val=%.4f", startEpoch, bestVal)

    for epoch in range(startEpoch, args.epochs + 1):
        t0 = time.time()
        trainMetrics = trainEpoch(
            model, trainLoader, optimizer, device,
            args.reconWeight, args.velWeight, args.commitWeight,
        )
        val_recon, util = validate(model, valLoader, device)
        scheduler.step()

        meanActive = sum(u["active_fraction"] for u in util) / max(len(util), 1)
        meanEntropyPct = (
            sum(u["entropy"] / max(u["max_entropy"], 1e-8) for u in util) / max(len(util), 1)
        )
        log.info(
            "[rvq] epoch %2d  loss=%.4f  recon=%.4f  vel=%.4f  val_recon=%.4f  "
            "cb_active=%.1f%%  cb_entropy=%.1f%%  (%.1fs)",
            epoch,
            trainMetrics.get("loss", 0.0),
            trainMetrics.get("recon", 0.0),
            trainMetrics.get("vel", 0.0),
            val_recon,
            meanActive * 100,
            meanEntropyPct * 100,
            time.time() - t0,
        )
        utilLog = {f"codebook/cb{i}_{k}": v for i, u in enumerate(util) for k, v in u.items()}
        wandbLog({
            "epoch": epoch,
            "train/loss": trainMetrics.get("loss", 0.0),
            "train/recon": trainMetrics.get("recon", 0.0),
            "train/vel": trainMetrics.get("vel", 0.0),
            "train/commit": trainMetrics.get("commit", 0.0),
            "val/recon": val_recon,
            "best_val_recon": min(bestVal, val_recon),
            "codebook/mean_active_pct": meanActive * 100,
            "codebook/mean_entropy_pct": meanEntropyPct * 100,
            **utilLog,
        }, step=epoch)

        ck = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_recon,
            "config": vars(args),
        }
        torch.save(ck, os.path.join(args.checkpointDir, f"checkpoint_epoch{epoch}.pt"))

        if val_recon < bestVal:
            bestVal = val_recon
            torch.save(ck, os.path.join(args.checkpointDir, "best_model.pt"))
            log.info("[rvq] best model saved (val_recon=%.4f)", val_recon)

    log.info("[rvq] done. best val_recon=%.4f", bestVal)

    # --- Final test evaluation on held-out 10% ---
    log.info("[rvq] loading best checkpoint for test evaluation...")
    bestCk = torch.load(
        os.path.join(args.checkpointDir, "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(bestCk["model_state_dict"])
    test_recon, test_util = validate(model, testLoader, device)
    testActive = sum(u["active_fraction"] for u in test_util) / max(len(test_util), 1)
    log.info(
        "[rvq] TEST recon=%.4f  cb_active=%.1f%%",
        test_recon, testActive * 100,
    )
    wandbLog({
        "test/recon": test_recon,
        "test/codebook_active_pct": testActive * 100,
    })

    wandbFinish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
