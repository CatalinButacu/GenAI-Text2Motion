"""Compute per-source translation (mean, std) for AMASS and HumanML3D.

Outputs an .npz with four arrays:
  amass_mean, amass_std       — translation channels 3:6 stats from AMASS-only
  humanml3d_mean, humanml3d_std — same from HumanML3D-only

Used by train_rvq_tokenizer.py via --trans-stats-path to fix the bimodal
translation distribution that hurt All-100's translation reconstruction
(MSE 0.73 — see Chapter 4 limitations).

Usage:
    python scripts/training/precompute_translation_stats.py \
        --output data/stats/translation_per_source.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from src.data.augmentation import detect_tpose, quality_filter, resample_to_fps
from src.data.dataset_cache import load_or_build_cache
from src.data.unified import build_sources_buffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig
from src.shared.constants import INGEST_MAX_LENGTH

log = logging.getLogger(__name__)

def compute_translation_stats(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over channels 3:6 across all frames in samples."""
    trans_frames = np.concatenate(
        [s["motion"][:, 3:6] for s in samples if s["motion"].shape[0] > 0], axis=0,
    )
    mean = trans_frames.mean(axis=0)
    std = trans_frames.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)

def compute_for_source(label: str, cfg: UnifiedConfig) -> tuple[np.ndarray, np.ndarray]:
    samples = build_sources_buffer(cfg, 30, resample_to_fps, quality_filter, detect_tpose,
                                  max_length=INGEST_MAX_LENGTH)

    if not samples:
        log.warning("[trans-stats] %s: 0 samples — falling back to (0, 1) stats", label)

        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    mean, std = compute_translation_stats(samples)
    log.info("[trans-stats]   %s mean|.|=%.4f  std|.|=%.4f  N=%d",
             label, float(np.abs(mean).mean()), float(np.abs(std).mean()), len(samples))

    return mean, std

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/AMASS", dest="data_dir")
    parser.add_argument("--humanml3d-dir", default="data/humanml3d", dest="humanml3d_dir")
    parser.add_argument("--arctic-dir", default="data/arctic/unpack", dest="arctic_dir")
    parser.add_argument("--interx-dir", default="data/inter-x", dest="interx_dir")
    parser.add_argument("--include-extras", action="store_true", dest="include_extras",
                        help="Also compute ARCTIC + InterX stats for --data mega runs.")
    parser.add_argument("--output", default="data/stats/translation_per_source.npz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    log.info("[trans-stats] AMASS source")
    amass_samples, _ = load_or_build_cache(args.data_dir, INGEST_MAX_LENGTH, None)
    amass_mean, amass_std = compute_translation_stats(amass_samples)
    log.info("[trans-stats]   amass mean|.|=%.4f  std|.|=%.4f  N=%d",
             float(np.abs(amass_mean).mean()), float(np.abs(amass_std).mean()), len(amass_samples))

    log.info("[trans-stats] HumanML3D source")
    human_cfg = UnifiedConfig(
        amass=SourceConfig(enabled=False),
        arctic=SourceConfig(enabled=False),
        humanml3d=SourceConfig(enabled=True, data_dir=args.humanml3d_dir, amass_dir=args.data_dir),
    )
    human_mean, human_std = compute_for_source("humanml3d", human_cfg)

    out: dict[str, np.ndarray] = {
        "amass_mean": amass_mean, "amass_std": amass_std,
        "humanml3d_mean": human_mean, "humanml3d_std": human_std,
    }

    if args.include_extras:
        log.info("[trans-stats] ARCTIC source")
        arctic_cfg = UnifiedConfig(
            amass=SourceConfig(enabled=False),
            arctic=SourceConfig(enabled=True, data_dir=args.arctic_dir),
            humanml3d=SourceConfig(enabled=False),
        )
        arctic_mean, arctic_std = compute_for_source("arctic", arctic_cfg)
        out["arctic_mean"] = arctic_mean
        out["arctic_std"] = arctic_std

        log.info("[trans-stats] InterX source")
        ix_cfg = UnifiedConfig(
            amass=SourceConfig(enabled=False),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=False),
            interx=SourceConfig(enabled=True, data_dir=args.interx_dir),
        )
        ix_mean, ix_std = compute_for_source("interx", ix_cfg)
        out["interx_mean"] = ix_mean
        out["interx_std"] = ix_std

    log.info("[trans-stats] amass std (m): %s", amass_std.tolist())
    log.info("[trans-stats] humanml3d std (m): %s", human_std.tolist())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, **out)
    log.info("[trans-stats] wrote %s with keys=%s", args.output, list(out.keys()))

    return 0

if __name__ == "__main__":
    sys.exit(main())
