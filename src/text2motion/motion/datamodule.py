from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from text2motion.motion.contracts import MotionDataConfig, Split
from text2motion.motion.datasets import (
    CaptionedMotionDataset,
    MotionWindowDataset,
    collate_motion_clips,
)
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM


class MotionDataModule:
    def __init__(
        self,
        root: Path,
        data: MotionDataConfig | None = None,
        texts_dir: Path | None = None,
        seed: int = 42,
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

    def captioned_batch_loader(
        self,
        split: Split | str,
        batch_size: int,
        shuffle: bool | None = None,
        num_workers: int = 0,
    ) -> DataLoader:
        split = Split(split)
        return DataLoader(
            self.captioned_dataset(split),
            batch_size=batch_size,
            shuffle=(split == Split.TRAIN) if shuffle is None else shuffle,
            num_workers=num_workers,
            collate_fn=collate_motion_clips,
            drop_last=(split == Split.TRAIN),
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
        )
