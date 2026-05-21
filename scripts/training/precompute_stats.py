"""Precompute z-score normalisation stats once, share across all RVQ runs.

Why this script exists:
    Each `train_rvq_tokenizer.py --data {amass,humanml3d,all}` run normally
    computes its own mean/std from its own training split. That makes
    val_recon numbers across the three runs incomparable -- HumanML3D's
    tighter variance produces a smaller std and a misleadingly smaller MSE.

    For the dissertation ablation table to be honest, all three runs must
    decode in the same normalised space. This script writes a single
    MotionStats payload to disk; the RVQ training script accepts it via
    --stats-path so every run uses identical (mean, std).

Default source for stats: AMASS-full (the broadest motion distribution).

Usage:
    python scripts/training/precompute_stats.py --source amass-full
    python scripts/training/precompute_stats.py --source unified
    python scripts/training/precompute_stats.py --output data/stats/custom.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from src.data.augmentation import detect_tpose, quality_filter, resample_to_fps
from src.data.dataset_cache import INGEST_MAX_LENGTH, load_or_build_cache
from src.data.motion_normalize import compute_motion_stats
from src.data.unified import build_sources_buffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig

log = logging.getLogger(__name__)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="amass-full",
                        choices=["amass-full", "humanml3d", "unified"],
                        help="Which corpus to compute stats from")
    parser.add_argument("--data-dir", default="data/AMASS", dest="data_dir")
    parser.add_argument("--humanml3d-dir", default="data/humanml3d", dest="humanml3d_dir")
    parser.add_argument("--max-motion-length", type=int, default=200, dest="max_motion_length")
    parser.add_argument("--output", default="data/stats/amass_full.npz",
                        help="Where to write the (mean, std) payload")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.source == "amass-full":
        log.info("[stats] computing from AMASS-full cache")
        samples, _ = load_or_build_cache(args.data_dir, INGEST_MAX_LENGTH, None)
    else:
        log.info("[stats] building unified buffer for source=%s", args.source)
        cfg = UnifiedConfig(
            amass=SourceConfig(enabled=(args.source == "unified"), data_dir=args.data_dir),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(
                enabled=True, data_dir=args.humanml3d_dir, amass_dir=args.data_dir,
            ),
        )
        samples = build_sources_buffer(cfg, 30, resample_to_fps, quality_filter, detect_tpose,
                                     max_length=INGEST_MAX_LENGTH)
    log.info("[stats] computing mean/std over %d samples", len(samples))
    stats = compute_motion_stats(samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, mean=stats.mean, std=stats.std)
    log.info("[stats] wrote %s  mean|.|=%.4f  std|.|=%.4f",
             args.output, float(np.abs(stats.mean).mean()), float(np.abs(stats.std).mean()))

    return 0

if __name__ == "__main__":
    sys.exit(main())
