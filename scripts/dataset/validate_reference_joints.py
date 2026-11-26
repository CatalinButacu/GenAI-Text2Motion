# .\.venv\Scripts\python.exe scripts/dataset/validate_reference_joints.py
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
VENDOR = REPO / "third_party" / "humanml3d_official"
FEATURES = REPO / "data" / "HumanML3D_official" / "new_joint_vecs" / "012314.npy"
TRUTH = REPO / "data" / "HumanML3D_official" / "validation" / "012314_official_joints.npy"
JOINTS = 22
PASS_MAX_ABS_M = 1e-4
PASS_BONE_DRIFT = 1e-4
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def load_recover():
    sys.path.insert(0, str(VENDOR))
    from recover import recover_from_ric

    return recover_from_ric


def bone_lengths(joints: np.ndarray) -> np.ndarray:
    lengths = []
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        lengths.append(np.linalg.norm(joints[:, child] - joints[:, parent], axis=-1))
    return np.stack(lengths, axis=-1)


def main() -> None:
    features = np.load(FEATURES).astype(np.float32)
    truth = np.load(TRUTH).astype(np.float32)
    if features.shape[1] != 263 or truth.shape[1:] != (JOINTS, 3):
        raise ValueError(f"unexpected shapes {features.shape} {truth.shape}")
    if features.shape[0] != truth.shape[0]:
        raise ValueError(f"frame count differs: {features.shape[0]} vs {truth.shape[0]}")
    recover_from_ric = load_recover()
    derived = recover_from_ric(torch.from_numpy(features), JOINTS).numpy()
    error = np.abs(derived - truth)
    max_abs = float(error.max())
    mean_abs = float(error.mean())
    bones_derived = bone_lengths(derived)
    bones_truth = bone_lengths(truth)
    bone_error = float(np.abs(bones_derived - bones_truth).max())
    drift = float((bones_derived.max(axis=0) - bones_derived.min(axis=0)).max())
    print("frames", features.shape[0])
    print("max abs error m", max_abs)
    print("mean abs error m", mean_abs)
    print("max bone length error m", bone_error)
    print("max bone length drift over time m", drift)
    passed = max_abs < PASS_MAX_ABS_M and drift < PASS_BONE_DRIFT
    print("RESULT", "PASS" if passed else "FAIL")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
