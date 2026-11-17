# .\.venv\Scripts\python.exe scripts/extra_motion/to_features.py aist [--lowpass_hz 6] [--fps 60]
import argparse
import csv
import pathlib
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
SOURCE_FPS = MappingProxyType(
    {
        "aist": 60,
        "style100": 60,
        "humman": 30,
        "motionx": 30,
        "motionx_official": 30,
        "amass_control": 20,
        "amass_extra": 20,
    }
)
TARGET_FPS_CHOICES = (20, 60)
LOWPASS_HZ = 8.0
LOWPASS_ORDER = 4
MAX_NON_FINITE_SHARE = 0.01
JOINT_COUNT = 22
MIN_SECONDS = 2.0
LEFT_ANKLE, LEFT_FOOT = 7, 10
SUMMARY_COLUMNS = (
    "source",
    "clips_in",
    "clips_written",
    "hours",
    "feet_forward_share",
    "non_finite",
)


def __excluded(source: str) -> set[str]:
    path = EXTRA / f"inventory_{source}.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["name"] for row in csv.DictReader(handle) if row["up_axis_ok"] != "True"}


def __to_target_fps(
    joints: np.ndarray, sos: np.ndarray | None, fps: int, target_fps: int
) -> np.ndarray:
    smooth = joints if sos is None else sosfiltfilt(sos, joints.astype(np.float64), axis=0)
    if fps == target_fps:
        return smooth
    if fps % target_fps == 0:
        return smooth[:: fps // target_fps]
    return resample_poly(smooth, target_fps, fps, axis=0, padtype="line")


def __lowpass(fps: int, cutoff: float | None) -> np.ndarray | None:
    if cutoff is None:
        return None
    return np.asarray(butter(LOWPASS_ORDER, cutoff, fs=fps, output="sos"))


def __feet_forward(features: np.ndarray) -> bool:
    joints = recover_from_ric(torch.from_numpy(features[:1]).float(), JOINT_COUNT)[0]
    return bool(joints[LEFT_FOOT, 2] - joints[LEFT_ANKLE, 2] > 0)


def __arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=sorted(SOURCE_FPS))
    parser.add_argument("--lowpass_hz", type=float, default=None)
    parser.add_argument("--fps", type=int, choices=TARGET_FPS_CHOICES, default=20)
    return parser.parse_args()


def main() -> None:
    arguments = __arguments()
    source = arguments.source
    target_fps = arguments.fps
    explicit_cutoff = arguments.lowpass_hz
    fps = SOURCE_FPS[source] if target_fps == 20 else target_fps
    joints_dir = EXTRA / f"joints{fps}" / source
    if not joints_dir.is_dir():
        raise FileNotFoundError(f"no joints folder {joints_dir}")
    reference = np.load(REFERENCE_JOINTS)[:, :JOINT_COUNT].reshape(-1, JOINT_COUNT, 3)
    params = feature_params(target_offsets(reference), target_fps)
    default_cutoff = None if fps == target_fps else LOWPASS_HZ
    sos = __lowpass(fps, default_cutoff if explicit_cutoff is None else explicit_cutoff)
    folder = source if explicit_cutoff is None else f"{source}_lp{explicit_cutoff:g}"
    out_dir = EXTRA / f"features{target_fps}" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded = __excluded(source)
    min_frames = round(MIN_SECONDS * target_fps)
    files = sorted(joints_dir.glob("*.npy"))
    written, frames, forward = 0, 0, 0
    non_finite = []
    for path in files:
        if path.stem in excluded:
            continue
        source_joints = np.load(path)
        if source_joints.shape[0] < MIN_SECONDS * fps:
            continue
        joints = __to_target_fps(source_joints, sos, fps, target_fps)
        if joints.shape[0] < min_frames:
            continue
        features = encode_joints(joints, params)[0].astype(np.float32)
        if not np.isfinite(features).all():
            non_finite.append(path.stem)
            continue
        np.save(out_dir / path.name, features)
        written += 1
        frames += features.shape[0]
        forward += int(__feet_forward(features))
    suffix = "" if target_fps == 20 else f"_{target_fps}fps"
    rejected = EXTRA / f"rejected_{folder}{suffix}.txt"
    rejected.write_text("".join(f"{name}\n" for name in non_finite), encoding="utf-8")
    if len(non_finite) > MAX_NON_FINITE_SHARE * len(files):
        raise ValueError(f"{len(non_finite)} of {len(files)} clips have non-finite features")
    row = (
        f"{folder}@{target_fps}fps",
        len(files),
        written,
        round(frames / target_fps / 3600, 2),
        round(forward / written, 3),
        len(non_finite),
    )
    print(dict(zip(SUMMARY_COLUMNS, row, strict=True)))


if __name__ == "__main__":
    main()
