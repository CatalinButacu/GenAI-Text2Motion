from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.dataset.training.batching import collate_motion_clips
from text2motion.dataset.training.captioned import CaptionedMotionDataset
from text2motion.dataset.training.windows import MixedWindowDataset, MotionWindowDataset
from text2motion.motion.contracts import (
    Split,
)
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM
from text2motion.motion.storage import (
    FEATURES_DIR,
    SPLIT_FILES,
    listed_clip_ids,
)

logger = logging.getLogger(__name__)


############################################################
#       MotionDataModule
############################################################
class MotionDataModule:
    def __init__(
        self,
        root: Path,
        data: MotionDataConfig | None = None,
        texts_dir: Path | None = None,
        *,
        seed: int,
    ) -> None:
        if root is None:
            raise ValueError("motion repository root must be set (regenerated 263 output dir)")
        self.root = Path(root)
        self.data = data or MotionDataConfig()
        self.texts_dir = Path(texts_dir) if texts_dir is not None else None
        self.seed = seed

    def load_scaler(self) -> MotionScaler:
        return MotionScaler.load(self.root, dim=DIM)

    def captioned_dataset(self, split: Split | str) -> CaptionedMotionDataset:
        return CaptionedMotionDataset(
            self.root, self.data, split=split, texts_dir=self.texts_dir, seed=self.seed
        )

    def window_dataset(self, split: Split | str, window: int) -> MotionWindowDataset:
        return MotionWindowDataset(self.root, self.data, split, window, seed=self.seed)

    def extension_window_dataset(
        self, extension_root: Path, window: int, scaler: MotionScaler
    ) -> MotionWindowDataset:
        vec_dir = Path(extension_root) / FEATURES_DIR
        names = listed_clip_ids(Path(extension_root) / SPLIT_FILES[Split.TRAIN])
        missing = [name for name in names if not (vec_dir / f"{name}.npy").is_file()]
        if not names or missing:
            raise FileNotFoundError(
                f"extension train list has {len(missing)} names without a feature file under {vec_dir}"
                f" (first: {missing[:3]}); the list is required so that no split leaks in"
            )
        return MotionWindowDataset(
            extension_root,
            self.data,
            Split.TRAIN,
            window,
            seed=self.seed,
            names=names,
            scaler=scaler,
        )

    def mixed_window_batch_loader(
        self,
        window: int,
        batch_size: int,
        extension_root: Path,
        share: float,
        num_workers: int = 0,
    ) -> DataLoader:
        primary = self.window_dataset(Split.TRAIN, window)
        extension = self.extension_window_dataset(extension_root, window, primary.scaler)
        return DataLoader(
            MixedWindowDataset(primary, extension, share, seed=self.seed),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_motion_clips,
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def captioned_batch_loader(
        self,
        split: Split | str,
        batch_size: int,
        shuffle: bool | None = None,
        num_workers: int = 0,
    ) -> DataLoader:
        split = Split(split)
        dataset = self.captioned_dataset(split)
        shuffled = (split == Split.TRAIN) if shuffle is None else shuffle
        weights = dataset.clip_weights if shuffled else None
        sampler = None if weights is None else self.weighted_sampler(weights)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffled and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_motion_clips,
            drop_last=(split == Split.TRAIN),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def weighted_sampler(self, weights: np.ndarray) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights.tolist(),
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def window_batch_loader(
        self,
        split: Split | str,
        window: int,
        batch_size: int,
        num_workers: int = 0,
    ) -> DataLoader:
        split = Split(split)
        return DataLoader(
            self.window_dataset(split, window),
            batch_size=batch_size,
            shuffle=(split == Split.TRAIN),
            num_workers=num_workers,
            collate_fn=collate_motion_clips,
            drop_last=(split == Split.TRAIN),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
