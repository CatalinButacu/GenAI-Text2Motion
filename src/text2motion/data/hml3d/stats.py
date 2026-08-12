from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .feature import normalization_stats


@dataclass(frozen=True)
class MotionScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(cls, out_dir: Path, dim: int | None = None) -> MotionScaler:
        mean_path, std_path = out_dir / "Mean.npy", out_dir / "Std.npy"
        if not mean_path.is_file() or not std_path.is_file():
            raise FileNotFoundError(
                f"Mean.npy/Std.npy missing under {out_dir}. For the standard 263 track these ship "
                f"with the official HumanML3D conversion; regenerating them changes the scaler "
                f"every trained checkpoint depends on."
            )
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        if mean.shape != std.shape:
            raise ValueError(f"Mean/Std shape mismatch: {mean.shape} vs {std.shape}")
        if dim is not None and mean.shape[-1] != dim:
            raise ValueError(f"Mean/Std must be {dim}-dim, got {mean.shape}")
        return cls(mean=mean, std=std)

    def normalize(self, feats):
        if isinstance(feats, torch.Tensor):
            return (feats - self._as_tensor(self.mean, feats)) / self._as_tensor(self.std, feats)
        return (feats - self.mean) / self.std

    def denormalize(self, feats):
        if isinstance(feats, torch.Tensor):
            return feats * self._as_tensor(self.std, feats) + self._as_tensor(self.mean, feats)
        return feats * self.std + self.mean

    @staticmethod
    def _as_tensor(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(array, device=like.device, dtype=like.dtype)


def read_split_names(split_path: Path) -> list[str]:
    if not split_path.is_file():
        raise FileNotFoundError(
            f"{split_path} missing: normalization statistics must be fitted on the train split "
            f"alone. Fitting over every available feature file would include val and test, "
            f"leaking held-out data into the scaler used by every downstream model."
        )
    return [n.strip() for n in split_path.read_text().splitlines() if n.strip()]


def load_clips(vec_dir: Path, names: list[str]) -> list[np.ndarray]:
    clips = []
    for name in sorted(set(names)):
        path = vec_dir / f"{name}.npy"
        if not path.is_file():
            continue
        arr = np.load(path)
        if np.isnan(arr).any():
            print(f"NaN feature, skipping: {name}")
            continue
        clips.append(arr)
    return clips


def fit_train_stats(
    vec_dir: Path, train_names: list[str], out_dir: Path, joints_num: int
) -> tuple[np.ndarray, np.ndarray]:
    clips = load_clips(vec_dir, train_names)
    mean, std = normalization_stats(clips, joints_num)

    mean_path = out_dir / "Mean.npy"
    provenance = {
        "fitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(vec_dir),
        "split": "train",
        "requested_clips": len(set(train_names)),
        "used_clips": len(clips),
        "frames": int(sum(c.shape[0] for c in clips)),
        "joints_num": joints_num,
        "overwrote_existing": mean_path.is_file(),
    }

    np.save(mean_path, mean)
    np.save(out_dir / "Std.npy", std)
    (out_dir / "stats_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Mean/Std saved from {len(clips)} train clips -> {out_dir / 'stats_provenance.json'}")
    return mean, std
