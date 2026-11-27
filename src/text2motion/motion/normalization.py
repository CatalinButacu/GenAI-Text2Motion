from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from text2motion.motion.contracts import Split
from text2motion.motion.storage import (
    MEAN_FILE,
    STATS_PROVENANCE_FILE,
    STD_FILE,
    atomic_numpy_save,
)

logger = logging.getLogger(__name__)


def check_scaler_values(mean: np.ndarray, std: np.ndarray, source: str | Path) -> None:
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(f"{source}: Mean/Std must be 1-D, got {mean.shape} and {std.shape}")
    non_finite = int((~np.isfinite(mean)).sum() + (~np.isfinite(std)).sum())
    if non_finite:
        raise ValueError(f"{source}: {non_finite} Mean/Std values are not finite")
    non_positive = int((std <= 0).sum())
    if non_positive:
        raise ValueError(
            f"{source}: {non_positive} Std values are zero or negative; normalization divides by them"
        )


############################################################
#       MotionScaler
############################################################
@dataclass(frozen=True)
class MotionScaler:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != self.std.shape:
            raise ValueError(f"Mean/Std shape mismatch: {self.mean.shape} vs {self.std.shape}")
        check_scaler_values(self.mean, self.std, "MotionScaler")

    def check_features(self, feats) -> None:
        floating = (
            feats.is_floating_point()
            if isinstance(feats, torch.Tensor)
            else np.issubdtype(np.asarray(feats).dtype, np.floating)
        )
        if not floating:
            raise TypeError("normalization needs floating-point features")
        if feats.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"features have width {feats.shape[-1]}, the scaler has {self.mean.shape[0]}"
            )

    @classmethod
    def load(cls, out_dir: Path, dim: int | None = None) -> MotionScaler:
        mean_path = out_dir / MEAN_FILE
        std_path = out_dir / STD_FILE
        if not mean_path.is_file() or not std_path.is_file():
            raise FileNotFoundError(
                f"Mean.npy/Std.npy missing under {out_dir}. For the standard 263 track these ship "
                f"with the official HumanML3D conversion; regenerating them changes the scaler "
                f"every trained checkpoint depends on."
            )
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        if dim is not None and mean.shape[-1] != dim:
            raise ValueError(f"Mean/Std must be {dim}-dim, got {mean.shape}")
        check_scaler_values(mean, std, out_dir)
        return cls(mean=mean, std=std)

    def normalize(self, feats):
        self.check_features(feats)
        if isinstance(feats, torch.Tensor):
            return (feats - self.as_tensor(self.mean, feats)) / self.as_tensor(self.std, feats)
        return (feats - self.mean) / self.std

    def denormalize(self, feats):
        self.check_features(feats)
        if isinstance(feats, torch.Tensor):
            return feats * self.as_tensor(self.std, feats) + self.as_tensor(self.mean, feats)
        return feats * self.std + self.mean

    @staticmethod
    def as_tensor(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(array, device=like.device, dtype=like.dtype)


def require_clip_files(vec_dir: Path, unique: list[str]) -> None:
    missing = [name for name in unique if not (vec_dir / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} train clips have no feature file under {vec_dir}: {missing[:10]}"
        )


def accumulate_clip_moments(clips: Iterable[np.ndarray]) -> tuple[int, np.ndarray, np.ndarray]:
    count = 0
    mean = np.zeros(0)
    squares = np.zeros(0)
    for clip in clips:
        data = np.asarray(clip, dtype=np.float64)
        frames = data.shape[0]
        clip_mean = data.mean(axis=0)
        clip_squares = ((data - clip_mean) ** 2).sum(axis=0)
        if count == 0:
            count, mean, squares = frames, clip_mean, clip_squares
            continue
        total = count + frames
        delta = clip_mean - mean
        mean = mean + delta * frames / total
        squares = squares + clip_squares + delta**2 * count * frames / total
        count = total
    return count, mean, squares


def expected_feature_dim(joint_count: int) -> int:
    return 8 + (joint_count - 1) * 9 + joint_count * 3


def check_feature_dim(std: np.ndarray, joint_count: int) -> None:
    expected_dim = expected_feature_dim(joint_count)
    if std.shape[-1] != expected_dim:
        raise ValueError(
            f"feature dim {std.shape[-1]} != expected {expected_dim} for {joint_count=}"
        )


def feature_groups(joint_count: int) -> tuple[tuple[int, int], ...]:
    ric_end = 4 + (joint_count - 1) * 3
    rot_end = 4 + (joint_count - 1) * 9
    vel_end = rot_end + joint_count * 3
    return (
        (0, 1),
        (1, 3),
        (3, 4),
        (4, ric_end),
        (ric_end, rot_end),
        (rot_end, vel_end),
        (vel_end, expected_feature_dim(joint_count)),
    )


def average_within_groups(std: np.ndarray, groups: tuple[tuple[int, int], ...]) -> None:
    for start, stop in groups:
        std[start:stop] = std[start:stop].mean()


def normalization_stats(
    clips: Iterable[np.ndarray], joint_count: int
) -> tuple[np.ndarray, np.ndarray]:
    count, mean, squares = accumulate_clip_moments(clips)
    if count == 0:
        raise ValueError("normalization_stats received no clips")

    mean = mean.astype(np.float32)
    std = np.sqrt(squares / count).astype(np.float32)

    check_feature_dim(std, joint_count)
    average_within_groups(std, feature_groups(joint_count))
    return mean, std


def streamed_clips(vec_dir: Path, unique: list[str]) -> Iterator[np.ndarray]:
    return (load_checked_features(vec_dir, name) for name in unique)


def load_checked_features(vec_dir: Path, name: str) -> np.ndarray:
    clip = np.load(vec_dir / f"{name}.npy")
    if clip.ndim != 2 or clip.shape[0] == 0 or not np.isfinite(clip).all():
        raise ValueError(f"train clip {name} is empty, not 2-D, or holds NaN or Inf")
    return clip


def count_frames(vec_dir: Path, unique: list[str]) -> int:
    frame_counts = [np.load(vec_dir / f"{name}.npy", mmap_mode="r").shape[0] for name in unique]
    return int(sum(frame_counts))


def fit_provenance(
    vec_dir: Path,
    train_names: list[str],
    unique: list[str],
    joint_count: int,
    frames: int,
    overwrote_existing: bool,
) -> dict[str, object]:
    return {
        "fitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(vec_dir),
        "split": Split.TRAIN.value,
        "requested_clips": len(set(train_names)),
        "used_clips": len(unique),
        "frames": frames,
        "joints_num": joint_count,
        "overwrote_existing": overwrote_existing,
    }


def fit_train_stats(
    vec_dir: Path, train_names: list[str], out_dir: Path, joint_count: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(train_names))
    require_clip_files(vec_dir, unique)
    mean, std = normalization_stats(streamed_clips(vec_dir, unique), joint_count)
    frames = count_frames(vec_dir, unique)

    mean_path = out_dir / MEAN_FILE
    provenance = fit_provenance(
        vec_dir, train_names, unique, joint_count, frames, mean_path.is_file()
    )

    check_scaler_values(mean, std, vec_dir)
    atomic_numpy_save(mean, mean_path)
    atomic_numpy_save(std, out_dir / STD_FILE)
    provenance_path = out_dir / STATS_PROVENANCE_FILE
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info(f"Mean/Std saved from {len(unique)} train clips -> {provenance_path}")
    return mean, std
