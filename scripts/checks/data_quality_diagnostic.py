# python scripts/checks/data_quality_diagnostic.py --checkpoint runs/checkpoints/tokenizer/r1_fsq_g8
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_check import (
    FEATURE_DIR,
    JOINT_COUNT,
    absolute_mpjpe_mm,
    project_recover,
    root_aligned_mpjpe_mm,
)
from metric_on_checkpoint import load_module, reconstruct

from text2motion.dataset.contracts import QualityLimits
from text2motion.dataset.core.quality import quality_metrics
from text2motion.motion.normalization import MotionScaler

DATA_ROOT = Path("data/processed/humanml3d")
TEST_FILE = DATA_ROOT / "test.txt"
RESULT_FILE = Path("docs/project/data_quality_diagnostic.csv")
FRAME_RATE = 20
DOWNSAMPLE = 4
MIN_FRAMES = 40
METRIC_NAMES = ("skating", "floating_mm", "jerk", "yaw")


def clip_names(split_file: Path) -> list[str]:
    names = [
        line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return [name for name in names if not name.startswith("M")]


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    rank_first = np.argsort(np.argsort(first)).astype(np.float64)
    rank_second = np.argsort(np.argsort(second)).astype(np.float64)
    return float(np.corrcoef(rank_first, rank_second)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    module = load_module(args.checkpoint)
    scaler = MotionScaler.load(DATA_ROOT, dim=263)
    limits = QualityLimits()
    rows = []
    for name in clip_names(TEST_FILE):
        feature = np.load(FEATURE_DIR / f"{name}.npy").astype(np.float64)
        if feature.shape[0] < MIN_FRAMES:
            continue
        feature = feature[: (feature.shape[0] // DOWNSAMPLE) * DOWNSAMPLE]
        reference = project_recover(feature, JOINT_COUNT)
        reconstruction = project_recover(reconstruct(module, scaler, feature), JOINT_COUNT)
        skating, floating, jerk, yaw = quality_metrics(feature, FRAME_RATE, limits)
        rows.append(
            {
                "clip": name,
                "frames": feature.shape[0],
                "mpjpe_absolute_mm": absolute_mpjpe_mm(reference, reconstruction),
                "mpjpe_root_aligned_mm": root_aligned_mpjpe_mm(reference, reconstruction),
                "skating": skating,
                "floating_mm": floating,
                "jerk": jerk,
                "yaw": yaw,
            }
        )

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aligned = np.array([row["mpjpe_root_aligned_mm"] for row in rows])
    print(f"clips {len(rows)}  -> {RESULT_FILE}")
    for metric in METRIC_NAMES:
        values = np.array([row[metric] for row in rows], dtype=np.float64)
        print(f"spearman root-aligned vs {metric} {spearman(aligned, values):.3f}")


if __name__ == "__main__":
    main()
