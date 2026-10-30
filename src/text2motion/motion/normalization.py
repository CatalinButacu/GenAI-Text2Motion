from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from text2motion.motion.contracts import Split
from text2motion.motion.representation import normalization_stats
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
            return (feats - self.__as_tensor(self.mean, feats)) / self.__as_tensor(self.std, feats)
        return (feats - self.mean) / self.std

    def denormalize(self, feats):
        self.check_features(feats)
        if isinstance(feats, torch.Tensor):
            return feats * self.__as_tensor(self.std, feats) + self.__as_tensor(self.mean, feats)
        return feats * self.std + self.mean

    @staticmethod
    def __as_tensor(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(array, device=like.device, dtype=like.dtype)


def read_split_names(split_path: Path) -> list[str]:
    if not split_path.is_file():
        raise FileNotFoundError(
            f"{split_path} missing: normalization statistics must be fitted on the train split "
            f"alone. Fitting over every available feature file would include val and test, "
            f"leaking held-out data into the scaler used by every downstream model."
        )
    return [name.strip() for name in split_path.read_text().splitlines() if name.strip()]


def load_clips(vec_dir: Path, names: list[str]) -> list[np.ndarray]:
    unique = sorted(set(names))
    missing = [name for name in unique if not (vec_dir / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} train clips have no feature file under {vec_dir}: {missing[:10]}"
        )
    clips = [np.load(vec_dir / f"{name}.npy") for name in unique]
    malformed = [
        name
        for name, clip in zip(unique, clips, strict=True)
        if clip.ndim != 2 or clip.shape[0] == 0 or not np.isfinite(clip).all()
    ]
    if malformed:
        raise ValueError(
            f"{len(malformed)} train clips are empty, not 2-D, or hold NaN or Inf: {malformed[:10]}"
        )
    return clips


def fit_train_stats(
    vec_dir: Path, train_names: list[str], out_dir: Path, joint_count: int
) -> tuple[np.ndarray, np.ndarray]:
    clips = load_clips(vec_dir, train_names)
    mean, std = normalization_stats(clips, joint_count)

    mean_path = out_dir / MEAN_FILE
    provenance = {
        "fitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(vec_dir),
        "split": Split.TRAIN.value,
        "requested_clips": len(set(train_names)),
        "used_clips": len(clips),
        "frames": int(sum(clip.shape[0] for clip in clips)),
        "joints_num": joint_count,
        "overwrote_existing": mean_path.is_file(),
    }

    check_scaler_values(mean, std, vec_dir)
    atomic_numpy_save(mean, mean_path)
    atomic_numpy_save(std, out_dir / STD_FILE)
    provenance_path = out_dir / STATS_PROVENANCE_FILE
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info(f"Mean/Std saved from {len(clips)} train clips -> {provenance_path}")
    return mean, std
