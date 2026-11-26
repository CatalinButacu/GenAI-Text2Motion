# python scripts/checks/metric_check.py
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from text2motion.motion.recovery import recover_from_ric

FEATURE_DIR = Path("data/processed/humanml3d/new_joint_vecs")
SPLIT_FILE = Path("data/processed/humanml3d/val.txt")
RESULT_FILE = Path("docs/project/metric_check.csv")
JOINT_COUNT = 22
CLIP_COUNT = 20
MIN_FRAMES = 40
NOISE_SCALE = 0.01
NOISE_SEED = 2026


def quaternion_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (quat[..., index] for index in range(4))
    entries = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return entries.reshape(quat.shape[:-1] + (3, 3))


def conjugate(quat: np.ndarray) -> np.ndarray:
    return quat * np.array([1.0, -1.0, -1.0, -1.0])


def second_recover(feature: np.ndarray, joint_count: int) -> np.ndarray:
    frames = feature.shape[0]
    rot_vel = feature[:, 0].astype(np.float64)
    angle = np.concatenate([[0.0], np.cumsum(rot_vel[:-1])])
    zeros = np.zeros_like(angle)
    quat = np.stack([np.cos(angle), zeros, np.sin(angle), zeros], axis=-1)
    inverse = quaternion_matrix(conjugate(quat))

    step = np.zeros((frames, 3))
    step[1:, 0] = feature[:-1, 1]
    step[1:, 2] = feature[:-1, 2]
    root = np.cumsum(np.einsum("tij,tj->ti", inverse, step), axis=0)
    root[:, 1] = feature[:, 3]

    local = feature[:, 4 : (joint_count - 1) * 3 + 4].reshape(frames, joint_count - 1, 3)
    local = np.einsum("tij,tkj->tki", inverse, local.astype(np.float64))
    local[..., 0] += root[:, None, 0]
    local[..., 2] += root[:, None, 2]
    return np.concatenate([root[:, None, :], local], axis=1)


def project_recover(feature: np.ndarray, joint_count: int) -> np.ndarray:
    tensor = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
    return recover_from_ric(tensor, joint_count)[0].numpy().astype(np.float64)


def absolute_mpjpe_mm(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(np.linalg.norm(reference - reconstruction, axis=-1).mean() * 1000.0)


def root_aligned(joints: np.ndarray) -> np.ndarray:
    return joints - joints[:, :1, :]


def root_aligned_mpjpe_mm(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    return absolute_mpjpe_mm(root_aligned(reference), root_aligned(reconstruction))


def validation_names() -> list[str]:
    names = [line.strip() for line in SPLIT_FILE.read_text(encoding="utf-8").splitlines() if line]
    base = [name for name in names if not name.startswith("M")]
    selected = []
    for name in base:
        feature = np.load(FEATURE_DIR / f"{name}.npy", mmap_mode="r")
        if feature.shape[0] >= MIN_FRAMES:
            selected.append(name)
        if len(selected) == CLIP_COUNT:
            break
    if len(selected) < CLIP_COUNT:
        raise ValueError(f"only {len(selected)} validation clips have {MIN_FRAMES} frames")
    return selected


def main() -> None:
    rng = np.random.default_rng(NOISE_SEED)
    rows = []
    for name in validation_names():
        feature = np.load(FEATURE_DIR / f"{name}.npy").astype(np.float64)
        project = project_recover(feature, JOINT_COUNT)
        second = second_recover(feature, JOINT_COUNT)
        agreement = float(np.abs(project - second).max())

        noisy = feature + rng.normal(0.0, NOISE_SCALE, feature.shape)
        reference = second
        perturbed = second_recover(noisy, JOINT_COUNT)
        rows.append(
            {
                "clip": name,
                "frames": feature.shape[0],
                "max_abs_diff_project_vs_second_m": f"{agreement:.3e}",
                "mpjpe_absolute_mm": f"{absolute_mpjpe_mm(reference, perturbed):.2f}",
                "mpjpe_root_aligned_mm": f"{root_aligned_mpjpe_mm(reference, perturbed):.2f}",
            }
        )

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    worst = max(float(row["max_abs_diff_project_vs_second_m"]) for row in rows)
    print(f"clips {len(rows)}  worst project-vs-second difference {worst:.3e} m  -> {RESULT_FILE}")


if __name__ == "__main__":
    main()
