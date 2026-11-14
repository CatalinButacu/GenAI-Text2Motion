# .\.venv\Scripts\python.exe scripts/extra_motion/to_features.py aist
import csv
import pathlib
import sys
from types import MappingProxyType

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from text2motion.motion.representation import (
    encode_joints,
    feature_params,
    recover_from_ric,
    target_offsets,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
EXTRA = REPO / "data" / "extra_motion"
REFERENCE_JOINTS = REPO / "data" / "HumanML3D_263" / "joints" / "000021.npy"
SOURCE_FPS = MappingProxyType({"aist": 60, "style100": 60, "humman": 30})
TARGET_FPS = 20
LOWPASS_HZ = 8.0
LOWPASS_ORDER = 4
JOINT_COUNT = 22
MIN_FRAMES = 40
LEFT_ANKLE, LEFT_FOOT = 7, 10
SUMMARY_COLUMNS = ("source", "clips_in", "clips_written", "hours", "feet_forward_share")


def __excluded(source: str) -> set[str]:
    path = EXTRA / f"inventory_{source}.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["name"] for row in csv.DictReader(handle) if row["up_axis_ok"] != "True"}


def __to_20_fps(joints: np.ndarray, sos: np.ndarray, fps: int) -> np.ndarray:
    smooth = sosfiltfilt(sos, joints.astype(np.float64), axis=0)
    if fps % TARGET_FPS == 0:
        return smooth[:: fps // TARGET_FPS]
    return resample_poly(smooth, TARGET_FPS, fps, axis=0)


def __feet_forward(features: np.ndarray) -> bool:
    joints = recover_from_ric(torch.from_numpy(features[:1]).float(), JOINT_COUNT)[0]
    return bool(joints[LEFT_FOOT, 2] - joints[LEFT_ANKLE, 2] > 0)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) == 2 else ""
    if source not in SOURCE_FPS:
        raise SystemExit(f"usage: to_features.py <{'|'.join(SOURCE_FPS)}>")
    fps = SOURCE_FPS[source]
    joints_dir = EXTRA / f"joints{fps}" / source
    if not joints_dir.is_dir():
        raise FileNotFoundError(f"no joints folder {joints_dir}")
    reference = np.load(REFERENCE_JOINTS)[:, :JOINT_COUNT].reshape(-1, JOINT_COUNT, 3)
    params = feature_params(target_offsets(reference))
    sos = np.asarray(butter(LOWPASS_ORDER, LOWPASS_HZ, fs=fps, output="sos"))
    out_dir = EXTRA / "features20" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded = __excluded(source)
    files = sorted(joints_dir.glob("*.npy"))
    written, frames, forward = 0, 0, 0
    for path in files:
        if path.stem in excluded:
            continue
        joints = __to_20_fps(np.load(path), sos, fps)
        if joints.shape[0] < MIN_FRAMES:
            continue
        features = encode_joints(joints, params)[0].astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"{path.stem}: non-finite features")
        np.save(out_dir / path.name, features)
        written += 1
        frames += features.shape[0]
        forward += int(__feet_forward(features))
    row = (
        source,
        len(files),
        written,
        round(frames / TARGET_FPS / 3600, 2),
        round(forward / written, 3),
    )
    print(dict(zip(SUMMARY_COLUMNS, row, strict=True)))


if __name__ == "__main__":
    main()
