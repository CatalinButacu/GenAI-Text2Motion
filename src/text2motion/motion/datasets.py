from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from text2motion.motion.annotations import parse_text_file
from text2motion.motion.contracts import MotionDataConfig, Split, TextAnnotation
from text2motion.motion.model import MotionBatch, MotionClip
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM, FPS
from text2motion.motion.storage import (
    FEATURES_DIR,
    SPLIT_FILES,
    TEXTS_DIR,
    clip_frame_counts,
    item_rng,
)


class CaptionedMotionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        data: MotionDataConfig,
        split: Split | str = Split.TRAIN,
        texts_dir: Path | None = None,
        seed: int = 42,
    ) -> None:
        split = Split(split)
        self._min_len = data.min_motion_len
        self._max_len = data.max_motion_len
        self._train = split == Split.TRAIN
        self._mirror = data.mirror_augment and self._train
        self._seed = seed
        self._epoch = 0

        root = Path(root)
        self._vec_dir = root / FEATURES_DIR
        self._text_dir = Path(texts_dir) if texts_dir is not None else root / TEXTS_DIR
        self.scaler = MotionScaler.load(root, dim=DIM)

        split_path = root / SPLIT_FILES[split]
        if not split_path.is_file():
            raise FileNotFoundError(
                f"split file not found: {split_path} (copy the official {split} list there)"
            )
        listed = [n.strip() for n in split_path.read_text().splitlines() if n.strip()]
        base_names = list(dict.fromkeys(n[1:] if n.startswith("M") else n for n in listed))
        self._ids = self._select_usable_clip_ids(base_names)
        if not self._ids:
            raise RuntimeError(f"no usable clips for split {split!r} under {self._vec_dir}")

    @property
    def clip_ids(self) -> list[str]:
        return list(self._ids)

    @property
    def mean(self) -> np.ndarray:
        return self.scaler.mean

    @property
    def std(self) -> np.ndarray:
        return self.scaler.std

    def _clip_id_variants(self, name: str) -> list[str]:
        return [name, f"M{name}"] if self._mirror else [name]

    def _select_usable_clip_ids(self, names: list[str]) -> list[str]:
        candidates = [clip_id for name in names for clip_id in self._clip_id_variants(name)]
        lengths = clip_frame_counts(self._vec_dir, candidates)
        return [
            clip_id
            for name in names
            for clip_id in self._clip_id_variants(name)
            if lengths.get(clip_id, 0) >= self._min_len
            and (self._text_dir / f"{clip_id}.txt").is_file()
        ]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._ids)

    def normalize(self, feat):
        return self.scaler.normalize(feat)

    def denormalize(self, feat):
        return self.scaler.denormalize(feat)

    def __getitem__(self, idx: int) -> MotionClip:
        clip_id = self._ids[idx]
        feat = np.load(self._vec_dir / f"{clip_id}.npy").astype(np.float32)
        if feat.shape[-1] != DIM:
            raise ValueError(f"{clip_id}: expected {DIM}-dim feature, got {feat.shape}")
        if not np.isfinite(feat).all():
            raise ValueError(f"{clip_id}: non-finite values in feature; drop it from the split list")

        rng = item_rng(self._seed, self._epoch, idx)
        annotations = parse_text_file(self._text_dir / f"{clip_id}.txt")
        annotation, feat = self._select_captioned_segment(annotations, feat, rng)
        feat = self.normalize(self._crop_to_max_frames(feat, rng))
        return MotionClip(
            features=torch.from_numpy(feat),
            frame_count=feat.shape[0],
            caption=annotation.caption,
            clip_id=clip_id,
        )

    @staticmethod
    def _annotation_frame_range(annotation: TextAnnotation, total: int) -> tuple[int, int] | None:
        if annotation.start_time == 0.0 and annotation.end_time == 0.0:
            return None
        start = max(int(annotation.start_time * FPS), 0)
        end = min(int(annotation.end_time * FPS), total)
        return start, end

    def _select_captioned_segment(
        self, annotations: list[TextAnnotation], feat: np.ndarray, rng: np.random.RandomState
    ) -> tuple[TextAnnotation, np.ndarray]:
        usable = []
        for annotation in annotations:
            segment = self._annotation_frame_range(annotation, feat.shape[0])
            if segment is None or segment[1] - segment[0] >= self._min_len:
                usable.append((annotation, segment))
        if not usable:
            return annotations[rng.randint(len(annotations))], feat
        annotation, segment = usable[rng.randint(len(usable))]
        return annotation, feat if segment is None else feat[segment[0] : segment[1]]

    def _crop_to_max_frames(self, feat: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        overflow = feat.shape[0] - self._max_len
        if overflow <= 0:
            return feat
        start = int(rng.randint(overflow + 1)) if self._train else 0
        return feat[start : start + self._max_len]


class MotionWindowDataset(Dataset):
    def __init__(
        self,
        root: Path,
        data: MotionDataConfig,
        split: Split | str,
        window: int,
        seed: int = 42,
    ) -> None:
        split = Split(split)
        self._window = window
        self._train = split == Split.TRAIN
        self._mirror = data.mirror_augment and self._train
        self._seed = seed
        self._epoch = 0

        root = Path(root)
        self._vec_dir = root / FEATURES_DIR
        self.scaler = MotionScaler.load(root, dim=DIM)
        listed = [
            n.strip() for n in (root / SPLIT_FILES[split]).read_text().splitlines() if n.strip()
        ]
        base_names = list(dict.fromkeys(n[1:] if n.startswith("M") else n for n in listed))
        self._ids, self._cache = self._load_windowable_clips(base_names)
        if not self._ids:
            raise RuntimeError(f"no clips >= window {window} for split {split!r}")

    def _load_windowable_clips(self, names: list[str]) -> tuple[list[str], list[np.ndarray]]:
        ids: list[str] = []
        clips: list[np.ndarray] = []
        non_finite: list[str] = []
        for name in names:
            variants = [name, f"M{name}"] if self._mirror else [name]
            for clip_id in variants:
                vec_path = self._vec_dir / f"{clip_id}.npy"
                if not vec_path.is_file():
                    continue
                feat = np.load(vec_path).astype(np.float32)
                if feat.shape[0] < self._window:
                    continue
                if not np.isfinite(feat).all():
                    non_finite.append(clip_id)
                    continue
                ids.append(clip_id)
                clips.append(feat)
        if non_finite:
            suffix = " ..." if len(non_finite) > 20 else ""
            raise ValueError(
                f"{len(non_finite)} non-finite clips in this split; drop them from the split list: "
                f"{non_finite[:20]}{suffix}"
            )
        return ids, clips

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

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

    def __getitem__(self, idx: int) -> MotionClip:
        feat = self._cache[idx]
        overflow = feat.shape[0] - self._window
        if self._train:
            start = int(item_rng(self._seed, self._epoch, idx).randint(overflow + 1))
        else:
            start = overflow // 2
        window = self.scaler.normalize(feat[start : start + self._window])
        return MotionClip(
            features=torch.from_numpy(window),
            frame_count=window.shape[0],
            clip_id=self._ids[idx],
        )


def collate_motion_clips(clips: list[MotionClip]) -> MotionBatch:
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
