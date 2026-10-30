from __future__ import annotations

from collections.abc import Sized
from multiprocessing import RawValue
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text2motion.motion.contracts import MotionDataConfig, Split, TextAnnotation
from text2motion.motion.model import MotionBatch, MotionClip
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM, FPS
from text2motion.motion.storage import (
    FEATURES_DIR,
    TEXTS_DIR,
    clip_frame_counts,
    item_rng,
    split_clip_ids,
)


def check_no_missing_files(kind: str, clip_ids: list[str]) -> None:
    if clip_ids:
        raise FileNotFoundError(
            f"{len(clip_ids)} clips of the split list have no {kind} file; first: {clip_ids[:5]}"
        )


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
        seed: int = 42,
    ) -> None:
        split = Split(split)
        self._min_len = data.min_motion_len
        self._max_len = data.max_motion_len
        self._train = split == Split.TRAIN
        self._mirror = data.mirror_augment and self._train
        self._seed = seed
        self._epoch = RawValue("q", 0)

        root = Path(root)
        self._vec_dir = root / FEATURES_DIR
        self._text_dir = Path(texts_dir) if texts_dir is not None else root / TEXTS_DIR
        self.scaler = MotionScaler.load(root, dim=DIM)

        self._ids = self.__select_usable_clip_ids(split_clip_ids(root, split))
        if not self._ids:
            raise RuntimeError(f"no usable clips for split {split!r} under {self._vec_dir}")
        uncaptioned = self.__clips_without_usable_caption()
        if uncaptioned:
            raise ValueError(
                f"{len(uncaptioned)} clips of split {split!r} have no caption that covers "
                f"{self._min_len} frames (first: {uncaptioned[:5]}); remove them from the split list"
            )

    @property
    def clip_ids(self) -> list[str]:
        return list(self._ids)

    @property
    def mean(self) -> np.ndarray:
        return self.scaler.mean

    @property
    def std(self) -> np.ndarray:
        return self.scaler.std

    def __clip_id_variants(self, name: str) -> list[str]:
        return [name, f"M{name}"] if self._mirror else [name]

    def __select_usable_clip_ids(self, names: list[str]) -> list[str]:
        candidates = [clip_id for name in names for clip_id in self.__clip_id_variants(name)]
        lengths = clip_frame_counts(self._vec_dir, candidates)
        check_no_missing_files(
            "feature", [clip_id for clip_id in candidates if clip_id not in lengths]
        )
        check_no_missing_files(
            "caption",
            [c for c in candidates if not (self._text_dir / f"{c}.txt").is_file()],
        )
        return [clip_id for clip_id in candidates if lengths[clip_id] >= self._min_len]

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self._ids)

    def normalize(self, feat):
        return self.scaler.normalize(feat)

    def denormalize(self, feat):
        return self.scaler.denormalize(feat)

    def __getitem__(self, idx: int) -> MotionClip:
        if not -len(self._ids) <= idx < len(self._ids):
            raise IndexError(f"index {idx} is out of range for {len(self._ids)} clips")
        idx = idx % len(self._ids)
        clip_id = self._ids[idx]
        feat = np.load(self._vec_dir / f"{clip_id}.npy").astype(np.float32)
        if feat.shape[-1] != DIM:
            raise ValueError(f"{clip_id}: expected {DIM}-dim feature, got {feat.shape}")
        if not np.isfinite(feat).all():
            raise ValueError(
                f"{clip_id}: non-finite values in feature; drop it from the split list"
            )

        rng = item_rng(self._seed, self._epoch.value, idx)
        annotations = parse_text_file(self._text_dir / f"{clip_id}.txt")
        annotation, feat = self.__select_captioned_segment(annotations, feat, rng)
        feat = self.normalize(self.__crop_to_max_frames(feat, rng))
        return MotionClip(
            features=torch.from_numpy(feat),
            frame_count=feat.shape[0],
            caption=annotation.caption,
            clip_id=clip_id,
        )

    @staticmethod
    def __annotation_frame_range(annotation: TextAnnotation, total: int) -> tuple[int, int] | None:
        if annotation.start_time == 0.0 and annotation.end_time == 0.0:
            return None
        start = max(int(annotation.start_time * FPS), 0)
        end = min(int(annotation.end_time * FPS), total)
        return start, end

    def __usable_annotations(
        self, annotations: list[TextAnnotation], total: int
    ) -> list[tuple[TextAnnotation, tuple[int, int] | None]]:
        usable = []
        for annotation in annotations:
            segment = self.__annotation_frame_range(annotation, total)
            if segment is None or segment[1] - segment[0] >= self._min_len:
                usable.append((annotation, segment))
        return usable

    def __clips_without_usable_caption(self) -> list[str]:
        lengths = clip_frame_counts(self._vec_dir, self._ids)
        return [
            clip_id
            for clip_id in self._ids
            if not self.__usable_annotations(
                parse_text_file(self._text_dir / f"{clip_id}.txt"), lengths[clip_id]
            )
        ]

    def __select_captioned_segment(
        self, annotations: list[TextAnnotation], feat: np.ndarray, rng: np.random.RandomState
    ) -> tuple[TextAnnotation, np.ndarray]:
        usable = self.__usable_annotations(annotations, feat.shape[0])
        if not usable:
            raise ValueError("no caption of this clip covers a segment of the minimum length")
        annotation, segment = usable[rng.randint(len(usable))]
        return annotation, feat if segment is None else feat[segment[0] : segment[1]]

    def __crop_to_max_frames(self, feat: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        overflow = feat.shape[0] - self._max_len
        if overflow <= 0:
            return feat
        start = int(rng.randint(overflow + 1)) if self._train else 0
        return feat[start : start + self._max_len]


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
        seed: int = 42,
    ) -> None:
        split = Split(split)
        if window <= 0:
            raise ValueError(f"the window must be a positive frame count, got {window}")
        self._window = window
        self._train = split == Split.TRAIN
        self._mirror = data.mirror_augment and self._train
        self._seed = seed
        self._epoch = RawValue("q", 0)

        root = Path(root)
        self._vec_dir = root / FEATURES_DIR
        self.scaler = MotionScaler.load(root, dim=DIM)
        self._ids, self._cache = self.__load_windowable_clips(split_clip_ids(root, split))
        if not self._ids:
            raise RuntimeError(f"no clips >= window {window} for split {split!r}")
        self._overflows = np.array([clip.shape[0] - window for clip in self._cache], np.int64)
        self._starts = self._overflows // 2
        self._starts_epoch = -1

    def __load_windowable_clips(self, names: list[str]) -> tuple[list[str], list[np.ndarray]]:
        ids: list[str] = []
        clips: list[np.ndarray] = []
        non_finite: list[str] = []
        variants = [v for name in names for v in ([name, f"M{name}"] if self._mirror else [name])]
        check_no_missing_files(
            "feature", [v for v in variants if not (self._vec_dir / f"{v}.npy").is_file()]
        )
        for clip_id in variants:
            feat = np.load(self._vec_dir / f"{clip_id}.npy").astype(np.float32)
            if feat.shape[0] < self._window:
                continue
            if not np.isfinite(feat).all():
                non_finite.append(clip_id)
                continue
            ids.append(clip_id)
            clips.append(np.asarray(self.scaler.normalize(feat), np.float32))
        if non_finite:
            suffix = " ..." if len(non_finite) > 20 else ""
            raise ValueError(
                f"{len(non_finite)} non-finite clips in this split; drop them from the split list: "
                f"{non_finite[:20]}{suffix}"
            )
        return ids, clips

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
        window = self._cache[idx][start : start + self._window]
        return MotionClip(
            features=torch.from_numpy(window),
            frame_count=window.shape[0],
            clip_id=self._ids[idx],
        )


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


def parse_text_file(path: Path) -> list[TextAnnotation]:
    annotations: list[TextAnnotation] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            raise ValueError(f"malformed caption line in {path}: {raw_line!r}")
        caption, tokens, start, end = parts[0], parts[1], parts[2], parts[3]
        start_time = float(start)
        end_time = float(end)
        annotations.append(
            TextAnnotation(
                caption=caption,
                tokens=tokens.split(" ") if tokens else [],
                start_time=0.0 if np.isnan(start_time) else start_time,
                end_time=0.0 if np.isnan(end_time) else end_time,
            )
        )
    if not annotations:
        raise ValueError(f"no captions found in {path}")
    return annotations


############################################################
#       MotionDataModule
############################################################
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
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
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
