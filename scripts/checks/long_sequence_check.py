# python scripts/checks/long_sequence_check.py
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metric_on_checkpoint as moc
from metric_check import JOINT_COUNT, project_recover, root_aligned

from text2motion.motion.normalization import MotionScaler

LONG_FEATURES = Path("data/processed/amass_263/new_joint_vecs")
DATA_ROOT = Path("data/processed/humanml3d")
RESULT_FILE = Path("docs/project/long_sequence_check.csv")
MODELS = (
    (
        "non_causal_e500",
        "configs/tokenizer/width_sweep/fsq_g8_w1024.yaml",
        "runs/checkpoints/tokenizer/r7_fsq_w1024_e500",
    ),
    (
        "causal_e100",
        "configs/tokenizer/causal/fsq_g8_w1024_causal.yaml",
        "runs/checkpoints/tokenizer/r8_causal_fsq_w1024_e100",
    ),
    (
        "causal_w256_e100",
        "configs/tokenizer/causal/fsq_g8_w1024_causal.yaml",
        "runs/checkpoints/tokenizer/r9_causal_w256_e100",
    ),
)
MIN_FRAMES = 800
CLIP_COUNT = 10
BIN_FRAMES = 200
DOWNSAMPLE = 4


def long_clip_names() -> list[str]:
    names = []
    for path in sorted(LONG_FEATURES.glob("*.npy")):
        if np.load(path, mmap_mode="r").shape[0] >= MIN_FRAMES:
            names.append(path.stem)
        if len(names) == CLIP_COUNT:
            break
    if len(names) < CLIP_COUNT:
        raise ValueError(f"only {len(names)} long clips with {MIN_FRAMES} frames")
    return names


def per_frame_error(reference: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    return np.linalg.norm(reference - reconstruction, axis=-1).mean(axis=-1) * 1000.0


def binned(values: np.ndarray) -> list[float]:
    count = values.shape[0] // BIN_FRAMES
    return [
        float(values[index * BIN_FRAMES : (index + 1) * BIN_FRAMES].mean())
        for index in range(count)
    ]


def main() -> None:
    scaler = MotionScaler.load(DATA_ROOT, dim=263)
    names = long_clip_names()
    rows = []
    for label, config, checkpoint in MODELS:
        moc.CONFIG = Path(config)
        module = moc.load_module(Path(checkpoint))
        absolute_bins, aligned_bins = [], []
        for name in names:
            feature = np.load(LONG_FEATURES / f"{name}.npy").astype(np.float64)
            feature = feature[: (feature.shape[0] // DOWNSAMPLE) * DOWNSAMPLE]
            reference = project_recover(feature, JOINT_COUNT)
            reconstruction = project_recover(moc.reconstruct(module, scaler, feature), JOINT_COUNT)
            absolute_bins.append(binned(per_frame_error(reference, reconstruction)))
            aligned_bins.append(
                binned(per_frame_error(root_aligned(reference), root_aligned(reconstruction)))
            )
        common = min(len(bins) for bins in absolute_bins)
        absolute = np.mean(np.array([bins[:common] for bins in absolute_bins]), axis=0)
        aligned = np.mean(np.array([bins[:common] for bins in aligned_bins]), axis=0)
        for index, (a, b) in enumerate(zip(absolute, aligned, strict=True)):
            rows.append(
                {
                    "model": label,
                    "bin_start_frame": index * BIN_FRAMES,
                    "absolute_mpjpe_mm": f"{a:.2f}",
                    "root_aligned_mpjpe_mm": f"{b:.2f}",
                }
            )
        print(
            f"{label}: first bin absolute {absolute[0]:.1f} mm, last bin absolute {absolute[-1]:.1f} mm, root-aligned first {aligned[0]:.1f} mm, last {aligned[-1]:.1f} mm"
        )

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"clips {len(names)}  -> {RESULT_FILE}")


if __name__ == "__main__":
    main()
