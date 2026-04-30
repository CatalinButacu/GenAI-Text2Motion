from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


@dataclass(slots=True)
class MotionStats:
    """Per-channel mean/std for SMPL-X motion tensors.

    Computed once over the training split and saved to the checkpoint so
    inference applies the exact same normalisation.
    """
    mean: np.ndarray   # (motion_dim,) float32
    std: np.ndarray    # (motion_dim,) float32

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.asarray(self.std, dtype=np.float32)
        assert self.mean.shape == self.std.shape and self.mean.ndim == 1


def computeMotionStats(samples: list[dict], key: str = "motion") -> MotionStats:
    """Compute per-channel mean/std across all frames of all samples.

    Zero-variance channels (jaw/eyes in AMASS) are kept with std=1 so
    normalisation becomes a no-op for them.
    """
    if not samples:
        raise ValueError("compute_motion_stats: empty sample list")
    frames = np.concatenate([s[key] for s in samples if s[key].shape[0] > 0], axis=0)
    mean = frames.mean(axis=0)
    std = frames.std(axis=0)
    std = np.where(std < EPS, 1.0, std)

    return MotionStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def normalize(motion: np.ndarray, stats: MotionStats,
              clipValue: float | None = 5.0) -> np.ndarray:
    """Z-score normalise per channel, then clip to ±clipValue sigma.

    Clipping caps damage from rare outlier samples that would otherwise dominate
    MSE loss. ±5σ keeps 99.99994% of N(0,1) data unaffected -- only true tail
    samples are bounded.
    """
    z = (motion - stats.mean) / stats.std

    if clipValue is not None:
        z = np.clip(z, -clipValue, clipValue)

    return z.astype(np.float32)


def denormalize(motion: np.ndarray, stats: MotionStats) -> np.ndarray:
    return (motion * stats.std + stats.mean).astype(np.float32)
