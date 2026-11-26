from __future__ import annotations

from multiprocessing import RawValue
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.dataset.training.batching import item_rng
from text2motion.dataset.training.captions import CaptionPicker
from text2motion.dataset.training.clip_index import ClipIndex, check_no_missing_files
from text2motion.dataset.training.sampling import (
    TrainWeighting,
    crop_to_max_frames,
    train_weighting,
)
from text2motion.motion.contracts import (
    MotionClip,
    Split,
    TextAnnotation,
)
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM

__all__ = ("CaptionedMotionDataset", "check_no_missing_files")


############################################################
#       CaptionedMotionDataset
############################################################
class CaptionedMotionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        data: MotionDataConfig,
        split: Split | str = Split.TRAIN,
        texts_dir: Path | None = None,
        *,
        seed: int,
    ) -> None:
        split = Split(split)
        self._max_len = data.max_motion_len
        self._train = split == Split.TRAIN
        self._seed = seed
        self._epoch = RawValue("q", 0)

        root = Path(root)
        self.scaler = MotionScaler.load(root, dim=DIM)
        self._index = ClipIndex(root, data, split, texts_dir)
        self.dropped_clip_ids = self._index.dropped_clip_ids

        weighting = train_weighting(self._index, data) if self._train else TrainWeighting()
        self._clip_weights = weighting.clip_weights
        self._picker = CaptionPicker(
            data.min_motion_len, weighting.verb_clip_counts, weighting.verb_group_sizes
        )

    @property
    def verb_clip_counts(self) -> dict[str, int] | None:
        return self._picker.verb_clip_counts

    @property
    def clip_weights(self) -> np.ndarray | None:
        return None if self._clip_weights is None else self._clip_weights.copy()

    @property
    def clip_ids(self) -> list[str]:
        return list(self._index.ids)

    def captions(self) -> list[str]:
        return sorted(
            {
                annotation.caption
                for annotations in self._index.annotations.values()
                for annotation in annotations
            }
        )

    def caption_weights(self, annotations: list[TextAnnotation]) -> np.ndarray:
        return self._picker.caption_weights(annotations)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self._index.ids)

    def __getitem__(self, idx: int) -> MotionClip:
        count = len(self._index.ids)
        if not -count <= idx < count:
            raise IndexError(f"index {idx} is out of range for {count} clips")
        idx = idx % count
        clip_id = self._index.ids[idx]
        feat = self._index.load_features(clip_id)

        rng = item_rng(self._seed, self._epoch.value, idx)
        annotations = self._index.annotations[clip_id]
        annotation, feat = self._picker.select_captioned_segment(annotations, feat, rng)
        cropped = crop_to_max_frames(feat, self._max_len, self._train, rng)
        feat = self.scaler.normalize(cropped)
        return MotionClip(
            features=torch.from_numpy(feat),
            frame_count=feat.shape[0],
            caption=annotation.caption,
            clip_id=clip_id,
        )
