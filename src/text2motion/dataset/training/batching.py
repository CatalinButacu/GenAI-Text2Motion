from __future__ import annotations

import logging
from collections.abc import Sized

import numpy as np
import torch
from torch.utils.data import DataLoader

from text2motion.motion.contracts import (
    MotionBatch,
    MotionClip,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def dataset_size(loader: DataLoader) -> int:
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("the dataset of this loader has no length")
    return len(dataset)


def check_collatable(clips: list[MotionClip]) -> None:
    if not clips:
        raise ValueError("a motion batch needs at least one clip")
    first = clips[0].features
    for clip in clips:
        features = clip.features
        if features.dim() != 2 or not 0 < clip.frame_count == features.shape[0]:
            raise ValueError(
                f"clip {clip.clip_id!r}: features {tuple(features.shape)} do not hold "
                f"frame_count={clip.frame_count} frames"
            )
        if (features.shape[1], features.dtype, features.device) != (
            first.shape[1],
            first.dtype,
            first.device,
        ):
            raise ValueError(
                f"clip {clip.clip_id!r}: width, dtype or device differs from the first clip"
            )


def collate_motion_clips(clips: list[MotionClip]) -> MotionBatch:
    check_collatable(clips)
    max_len = max(clip.frame_count for clip in clips)
    feature_dim = clips[0].features.shape[1]
    padded = clips[0].features.new_zeros(len(clips), max_len, feature_dim)
    for row, clip in enumerate(clips):
        padded[row, : clip.frame_count] = clip.features
    return MotionBatch(
        features=padded,
        lengths=torch.tensor([clip.frame_count for clip in clips], dtype=torch.long),
        captions=[clip.caption or "" for clip in clips],
    )


def pad_to_window(clip: np.ndarray, window: int) -> np.ndarray:
    if clip.shape[0] >= window:
        return clip
    padded = np.zeros((window, clip.shape[1]), np.float32)
    padded[: clip.shape[0]] = clip
    return padded


def log_dropped_clips(dropped: list[str], reason: str) -> None:
    if dropped:
        logger.info(f"dropped {len(dropped)} clips {reason} (first: {dropped[:5]})")


def item_rng(seed: int, epoch: int, index: int) -> np.random.RandomState:
    item_seed = np.random.SeedSequence((seed, epoch, index)).generate_state(1)[0]
    return np.random.RandomState(item_seed)
