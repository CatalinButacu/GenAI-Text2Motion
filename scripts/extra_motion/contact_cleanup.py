# .\.venv\Scripts\python.exe scripts/extra_motion/contact_cleanup.py motionx_lp6 1000
import csv
import pathlib
import sys

import numpy as np
import torch
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from text2motion.motion.representation import (
    encode_joints,
    feature_params,
    recover_from_ric,
    target_offsets,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
EXTRA = REPO / "data" / "extra_motion"
REFERENCE_JOINTS = REPO / "data" / "HumanML3D_263" / "joints" / "000021.npy"
FPS = 20
JOINT_COUNT = 22
FEET = (7, 8, 10, 11)
FLOOR_MIN_SECONDS = 2
FLOOR_MEAN_SECONDS = 1
CONTACT_HEIGHT_M = 0.05
CONTACT_SPEED_M_PER_S = 1.0
SEED = 2026


def __kept_names(source: str) -> list[str]:
    with open(EXTRA / f"physics_{source}.csv", newline="", encoding="utf-8") as handle:
        return [
            row["name"]
            for row in csv.DictReader(handle)
            if row["matched_outlier"] != "True" and row["belt_walk"] != "True"
        ]


def __sample(names: list[str], count: int | None) -> list[str]:
    if count is None or count >= len(names):
        return names
    order = np.random.default_rng(SEED).choice(len(names), count, replace=False)
    return sorted(names[int(index)] for index in order)


def __level_floor(joints: np.ndarray) -> np.ndarray:
    lowest = joints[:, FEET, 1].min(axis=-1)
    floor = minimum_filter1d(lowest, size=FLOOR_MIN_SECONDS * FPS, mode="nearest")
    floor = uniform_filter1d(floor, size=FLOOR_MEAN_SECONDS * FPS, mode="nearest")
    leveled = joints.copy()
    leveled[..., 1] -= floor[:, None]
    return leveled


def __pin_contacts(joints: np.ndarray) -> np.ndarray:
    height = joints[:, FEET, 1]
    step = np.diff(joints[:, FEET][..., [0, 2]], axis=0)
    speed = np.linalg.norm(step, axis=-1) * FPS
    contact = (height[:-1] < CONTACT_HEIGHT_M) & (height[1:] < CONTACT_HEIGHT_M)
    contact &= speed < CONTACT_SPEED_M_PER_S
    count = contact.sum(axis=-1, keepdims=True)
    correction = -(step * contact[..., None]).sum(axis=1) / np.maximum(count, 1)
    offset = np.concatenate((np.zeros((1, 2)), np.cumsum(correction, axis=0)))
    pinned = joints.copy()
    pinned[..., [0, 2]] += offset[:, None, :]
    return pinned


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: contact_cleanup.py <source> [sample_count]")
    source = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) == 3 else None
    source_dir = EXTRA / "features20" / source
    folder = f"{source}_clean" if count is None else f"{source}_clean_s{count}"
    out_dir = EXTRA / "features20" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = np.load(REFERENCE_JOINTS)[:, :JOINT_COUNT].reshape(-1, JOINT_COUNT, 3)
    params = feature_params(target_offsets(reference))
    names = __sample(__kept_names(source), count)
    for name in names:
        features = torch.from_numpy(np.load(source_dir / f"{name}.npy")).float()
        joints = recover_from_ric(features, JOINT_COUNT).numpy().astype(np.float64)
        cleaned = __pin_contacts(__level_floor(joints))
        encoded = encode_joints(cleaned, params)[0].astype(np.float32)
        if not np.isfinite(encoded).all():
            raise ValueError(f"{name}: non-finite features after cleanup")
        np.save(out_dir / f"{name}.npy", encoded)
    print({"source": source, "folder": folder, "clips": len(names)})


if __name__ == "__main__":
    main()
