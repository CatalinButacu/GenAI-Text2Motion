from __future__ import annotations

import logging
from multiprocessing import RawValue
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.dataset.training.batching import log_dropped_clips, pad_to_window
from text2motion.dataset.training.clip_index import check_no_missing_files
from text2motion.motion.contracts import (
    MotionClip,
    Split,
)
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM
from text2motion.motion.storage import (
    FEATURES_DIR,
    base_clip_ids,
)

logger = logging.getLogger(__name__)


EXTENSION_PLAN_STREAM = 1


def require_positive_window(window: int) -> None:
    if window <= 0:
        raise ValueError(f"the window must be a positive frame count, got {window}")


def window_offsets(clips: list[np.ndarray], window: int) -> tuple[np.ndarray, np.ndarray]:
    overflows = np.array([max(clip.shape[0] - window, 0) for clip in clips], np.int64)
    return overflows, overflows // 2


def require_windowable_clips(ids: list[str], min_len: int, split: Split) -> None:
    if not ids:
        raise RuntimeError(f"no clips >= {min_len} frames for split {split!r}")


def expand_mirror_variants(names: list[str], mirror: bool) -> list[str]:
    return [v for name in names for v in ([name, f"M{name}"] if mirror else [name])]


def require_feature_files(vec_dir: Path, variants: list[str]) -> None:
    check_no_missing_files("feature", [v for v in variants if not (vec_dir / f"{v}.npy").is_file()])


def require_finite_clips(non_finite: list[str]) -> None:
    if non_finite:
        suffix = " ..." if len(non_finite) > 20 else ""
        raise ValueError(
            f"{len(non_finite)} non-finite clips in this split; drop them from the split list: "
            f"{non_finite[:20]}{suffix}"
        )


############################################################
#       MotionWindowDataset
############################################################
class MotionWindowDataset(Dataset):
    def __init__(
        self,
        root: Path,
        data: MotionDataConfig,
        split: Split | str,
        window: int,
        *,
        seed: int,
        names: list[str] | None = None,
        scaler: MotionScaler | None = None,
    ) -> None:
        split = Split(split)
        require_positive_window(window)
        self._window = window
        self._train = split == Split.TRAIN
        self._mirror = data.mirror_augment and self._train and names is None
        self._seed = seed
        self._epoch = RawValue("q", 0)
        root = Path(root)
        self._vec_dir = root / FEATURES_DIR
        self.scaler = MotionScaler.load(root, dim=DIM) if scaler is None else scaler
        self._min_len = data.min_motion_len
        clip_names = base_clip_ids(root, split) if names is None else names
        self._ids, self._cache, self._frames, self.dropped_clip_ids = self.load_windowable_clips(
            clip_names
        )
        log_dropped_clips(self.dropped_clip_ids, f"shorter than {self._min_len} frames")
        require_windowable_clips(self._ids, self._min_len, split)
        self._overflows, self._starts = window_offsets(self._cache, window)
        self._starts_epoch = -1

    def load_windowable_clips(
        self, names: list[str]
    ) -> tuple[list[str], list[np.ndarray], list[int], list[str]]:
        ids: list[str] = []
        short: list[str] = []
        clips: list[np.ndarray] = []
        frames: list[int] = []
        non_finite: list[str] = []
        variants = expand_mirror_variants(names, self._mirror)
        require_feature_files(self._vec_dir, variants)
        for clip_id in variants:
            feat = np.load(self._vec_dir / f"{clip_id}.npy").astype(np.float32)
            if feat.shape[0] < self._min_len:
                short.append(clip_id)
                continue
            if not np.isfinite(feat).all():
                non_finite.append(clip_id)
                continue
            normalized = np.asarray(self.scaler.normalize(feat), np.float32)
            ids.append(clip_id)
            frames.append(min(normalized.shape[0], self._window))
            clips.append(pad_to_window(normalized, self._window))
        require_finite_clips(non_finite)
        return ids, clips, frames, short

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def mean(self) -> np.ndarray:
        return self.scaler.mean

    @property
    def std(self) -> np.ndarray:
        return self.scaler.std

    def denormalize(self, feat: torch.Tensor) -> torch.Tensor:
        return self.scaler.denormalize(feat)

    def window_starts(self) -> np.ndarray:
        epoch = self._epoch.value
        if not self._train or epoch == self._starts_epoch:
            return self._starts
        generator = np.random.default_rng((self._seed, epoch))
        self._starts = generator.integers(self._overflows + 1)
        self._starts_epoch = epoch
        return self._starts

    def __getitem__(self, idx: int) -> MotionClip:
        start = int(self.window_starts()[idx])
        window = self._cache[idx][start : start + int(self._frames[idx])]
        return MotionClip(
            features=torch.from_numpy(window),
            frame_count=window.shape[0],
            clip_id=self._ids[idx],
        )


class MixedWindowDataset(Dataset):
    def __init__(
        self,
        primary: MotionWindowDataset,
        extension: MotionWindowDataset,
        share: float,
        *,
        seed: int,
    ) -> None:
        if not 0.0 <= share < 1.0:
            raise ValueError(f"the extension share must be in [0, 1), got {share}")
        self._primary = primary
        self._extension = extension
        self._share = share
        self._seed = seed
        self._plan: list[tuple[int, int]] = []
        self.set_epoch(0)

    @property
    def scaler(self) -> MotionScaler:
        return self._primary.scaler

    @property
    def extension_count(self) -> int:
        return round(len(self._primary) * self._share / (1.0 - self._share))

    def set_epoch(self, epoch: int) -> None:
        self._primary.set_epoch(epoch)
        self._extension.set_epoch(epoch)
        rng = np.random.default_rng((self._seed, epoch, EXTENSION_PLAN_STREAM))
        primary = [(0, index) for index in range(len(self._primary))]
        extension = rng.integers(len(self._extension), size=self.extension_count)
        self._plan = primary + [(1, int(index)) for index in extension]

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, index: int) -> MotionClip:
        source, position = self._plan[index]
        return (self._primary if source == 0 else self._extension)[position]
