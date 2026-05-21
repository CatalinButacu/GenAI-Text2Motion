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

from src.data.augmentation import detectTpose, qualityFilter, resampleToFps
from src.data.dataset_cache import INGEST_MAX_LENGTH, loadOrBuildCache
from src.data.unified import buildSourcesBuffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig

log = logging.getLogger(__name__)

def computeTranslationStats(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over channels 3:6 across all frames in samples."""
    transFrames = np.concatenate(
        [s["motion"][:, 3:6] for s in samples if s["motion"].shape[0] > 0], axis=0,
    )
    mean = transFrames.mean(axis=0)
    std = transFrames.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)

def computeForSource(label: str, cfg: UnifiedConfig) -> tuple[np.ndarray, np.ndarray]:
    samples = buildSourcesBuffer(cfg, 30, resampleToFps, qualityFilter, detectTpose,
                                  maxLength=INGEST_MAX_LENGTH)

    if not samples:
        log.warning("[trans-stats] %s: 0 samples — falling back to (0, 1) stats", label)

        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    mean, std = computeTranslationStats(samples)
    log.info("[trans-stats]   %s mean|.|=%.4f  std|.|=%.4f  N=%d",
             label, float(np.abs(mean).mean()), float(np.abs(std).mean()), len(samples))

    return mean, std

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/AMASS", dest="dataDir")
    parser.add_argument("--humanml3d-dir", default="data/humanml3d", dest="humanml3dDir")
    parser.add_argument("--arctic-dir", default="data/arctic/unpack", dest="arcticDir")
    parser.add_argument("--interx-dir", default="data/inter-x", dest="interxDir")
    parser.add_argument("--include-extras", action="store_true", dest="includeExtras",
                        help="Also compute ARCTIC + InterX stats for --data mega runs.")
    parser.add_argument("--output", default="data/stats/translation_per_source.npz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    log.info("[trans-stats] AMASS source")
    amassSamples, _ = loadOrBuildCache(args.dataDir, INGEST_MAX_LENGTH, None)
    amassMean, amassStd = computeTranslationStats(amassSamples)
    log.info("[trans-stats]   amass mean|.|=%.4f  std|.|=%.4f  N=%d",
             float(np.abs(amassMean).mean()), float(np.abs(amassStd).mean()), len(amassSamples))

    log.info("[trans-stats] HumanML3D source")
    humanCfg = UnifiedConfig(
        amass=SourceConfig(enabled=False),
        arctic=SourceConfig(enabled=False),
        humanml3d=SourceConfig(enabled=True, dataDir=args.humanml3dDir, amassDir=args.dataDir),
    )
    humanMean, humanStd = computeForSource("humanml3d", humanCfg)

    out: dict[str, np.ndarray] = {
        "amass_mean": amassMean, "amass_std": amassStd,
        "humanml3d_mean": humanMean, "humanml3d_std": humanStd,
    }

    if args.includeExtras:
        log.info("[trans-stats] ARCTIC source")
        arcticCfg = UnifiedConfig(
            amass=SourceConfig(enabled=False),
            arctic=SourceConfig(enabled=True, dataDir=args.arcticDir),
            humanml3d=SourceConfig(enabled=False),
        )
        arcticMean, arcticStd = computeForSource("arctic", arcticCfg)
        out["arctic_mean"] = arcticMean
        out["arctic_std"] = arcticStd

        log.info("[trans-stats] InterX source")
        ixCfg = UnifiedConfig(
            amass=SourceConfig(enabled=False),
            arctic=SourceConfig(enabled=False),
            humanml3d=SourceConfig(enabled=False),
            interx=SourceConfig(enabled=True, dataDir=args.interxDir),
        )
        ixMean, ixStd = computeForSource("interx", ixCfg)
        out["interx_mean"] = ixMean
        out["interx_std"] = ixStd

    log.info("[trans-stats] amass std (m): %s", amassStd.tolist())
    log.info("[trans-stats] humanml3d std (m): %s", humanStd.tolist())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, **out)
    log.info("[trans-stats] wrote %s with keys=%s", args.output, list(out.keys()))

    return 0

if __name__ == "__main__":
    sys.exit(main())
