from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sized
from multiprocessing import RawValue
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from text2motion.motion.contracts import (
    CaptionWeighting,
    ClipWeighting,
    MotionBatch,
    MotionClip,
    MotionDataConfig,
    Split,
    TextAnnotation,
)
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM, FPS
from text2motion.motion.storage import (
    FEATURES_DIR,
    MOTION_CLASS_POWER,
    NO_MOTION_CLASS,
    NO_VERB_GROUP,
    RARE_VERB_MIN_CLIPS,
    RARE_VERB_POWER,
    TEXTS_DIR,
    VERB_BALANCE_MIN_CLIPS,
    base_clip_ids,
    caption_verbs,
    clip_frame_counts,
    parse_text_file,
    read_motion_classes,
)

logger = logging.getLogger(__name__)


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
        *,
        seed: int,
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

        self._ids, self.dropped_clip_ids = self.__select_usable_clip_ids(base_clip_ids(root, split))
        log_dropped_clips(self.dropped_clip_ids, f"shorter than {self._min_len} frames")
        if not self._ids:
            raise RuntimeError(f"no usable clips for split {split!r} under {self._vec_dir}")
        uncaptioned = self.__clips_without_usable_caption()
        if uncaptioned:
            raise ValueError(
                f"{len(uncaptioned)} clips of split {split!r} have no caption that covers "
                f"{self._min_len} frames (first: {uncaptioned[:5]}); remove them from the split list"
            )
        self._verb_clip_counts: dict[str, int] | None = None
        if data.caption_weighting == CaptionWeighting.RARE_VERB and self._train:
            self._verb_clip_counts = self.__verb_clip_counts(
                RARE_VERB_MIN_CLIPS, "rare_verb caption"
            )
        self._verb_group_sizes: dict[str, int] | None = None
        self._clip_weights: np.ndarray | None = None
        if data.clip_weighting == ClipWeighting.MOTION_CLASS and self._train:
            if data.motion_classes is None:
                raise ValueError("motion_class clip weighting needs the motion_classes verb map")
            clip_classes = self.__clip_classes(read_motion_classes(data.motion_classes))
            self._clip_weights = self.__motion_class_weights(clip_classes)
        if data.clip_weighting == ClipWeighting.UNIFORM_REPLACEMENT and self._train:
            self._clip_weights = np.ones(len(self._ids), dtype=np.float64)
        if data.clip_weighting == ClipWeighting.VERB and self._train:
            balance_counts = self.__verb_clip_counts(VERB_BALANCE_MIN_CLIPS, "verb clip")
            self._verb_clip_counts = balance_counts
            caption_groups = self.__caption_groups(balance_counts)
            self._verb_group_sizes = self.__verb_group_sizes(caption_groups)
            self._clip_weights = self.__verb_group_weights(caption_groups, self._verb_group_sizes)

    def __clip_verbs(self) -> list[frozenset[str]]:
        found = []
        for clip_id in self._ids:
            annotations = parse_text_file(self._text_dir / f"{clip_id}.txt")
            found.append(frozenset().union(*(caption_verbs(a) for a in annotations)))
        return found

    def __clip_classes(self, classes: dict[str, int]) -> list[frozenset[int]]:
        found = []
        for verbs in self.__clip_verbs():
            labels = frozenset(classes[verb] for verb in verbs if verb in classes)
            found.append(labels or frozenset({NO_MOTION_CLASS}))
        return found

    @staticmethod
    def __rarest_verb(annotation: TextAnnotation, counts: dict[str, int]) -> str:
        kept = [verb for verb in caption_verbs(annotation) if verb in counts]
        return min(kept, key=lambda verb: (counts[verb], verb), default=NO_VERB_GROUP)

    def __caption_groups(self, counts: dict[str, int]) -> list[list[str]]:
        lengths = clip_frame_counts(self._vec_dir, self._ids)
        groups = []
        for clip_id in self._ids:
            annotations = parse_text_file(self._text_dir / f"{clip_id}.txt")
            usable = self.__usable_annotations(annotations, lengths[clip_id])
            groups.append([self.__rarest_verb(annotation, counts) for annotation, _ in usable])
        return groups

    @staticmethod
    def __verb_group_sizes(caption_groups: list[list[str]]) -> dict[str, int]:
        sizes = Counter(group for groups in caption_groups for group in groups)
        verb_sizes = [size for group, size in sizes.items() if group != NO_VERB_GROUP]
        sizes[NO_VERB_GROUP] = max(verb_sizes)
        return dict(sizes)

    @staticmethod
    def __verb_group_weights(caption_groups: list[list[str]], sizes: dict[str, int]) -> np.ndarray:
        weights = [sum(1.0 / sizes[group] for group in groups) for groups in caption_groups]
        return np.asarray(weights, dtype=np.float64)

    @staticmethod
    def __motion_class_weights(clip_classes: list[frozenset[int]]) -> np.ndarray:
        sizes = Counter(label for labels in clip_classes for label in labels)
        labelled = [size for label, size in sizes.items() if label != NO_MOTION_CLASS]
        if not labelled:
            raise ValueError("no train clip has a verb of the motion class map")
        sizes[NO_MOTION_CLASS] = max(labelled)
        rarest = [min(sizes[label] for label in labels) for labels in clip_classes]
        return np.asarray(rarest, dtype=np.float64) ** -MOTION_CLASS_POWER

    def __verb_clip_counts(self, min_clips: int, weighting: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for clip_id in self._ids:
            if clip_id.startswith("M"):
                continue
            annotations = parse_text_file(self._text_dir / f"{clip_id}.txt")
            for verb in frozenset().union(*(caption_verbs(a) for a in annotations)):
                counts[verb] = counts.get(verb, 0) + 1
        kept = {verb: n for verb, n in counts.items() if n >= min_clips}
        if not kept:
            raise ValueError(
                f"{weighting} weighting needs a verb in at least {min_clips} "
                "train clips; no verb has that many"
            )
        return kept

    @property
    def verb_clip_counts(self) -> dict[str, int] | None:
        return None if self._verb_clip_counts is None else dict(self._verb_clip_counts)

    @property
    def clip_weights(self) -> np.ndarray | None:
        return None if self._clip_weights is None else self._clip_weights.copy()

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

    def __select_usable_clip_ids(self, names: list[str]) -> tuple[list[str], list[str]]:
        candidates = [clip_id for name in names for clip_id in self.__clip_id_variants(name)]
        lengths = clip_frame_counts(self._vec_dir, candidates)
        check_no_missing_files(
            "feature", [clip_id for clip_id in candidates if clip_id not in lengths]
        )
        check_no_missing_files(
            "caption",
            [c for c in candidates if not (self._text_dir / f"{c}.txt").is_file()],
        )
        kept = [clip_id for clip_id in candidates if lengths[clip_id] >= self._min_len]
        dropped = [clip_id for clip_id in candidates if lengths[clip_id] < self._min_len]
        return kept, dropped

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
        annotation, segment = usable[self.__caption_index([a for a, _ in usable], rng)]
        return annotation, feat if segment is None else feat[segment[0] : segment[1]]

    def __caption_index(self, annotations: list[TextAnnotation], rng: np.random.RandomState) -> int:
        if self._verb_clip_counts is None:
            return rng.randint(len(annotations))
        return int(rng.choice(len(annotations), p=self.caption_weights(annotations)))

    def caption_weights(self, annotations: list[TextAnnotation]) -> np.ndarray:
        counts = self._verb_clip_counts
        if counts is None:
            return np.full(len(annotations), 1.0 / len(annotations))
        sizes = self._verb_group_sizes
        if sizes is not None:
            groups = [self.__rarest_verb(annotation, counts) for annotation in annotations]
            weights = np.asarray([1.0 / sizes[group] for group in groups], dtype=np.float64)
            return weights / weights.sum()
        common = max(counts.values())
        rarest = [
            min((counts[verb] for verb in caption_verbs(a) if verb in counts), default=common)
            for a in annotations
        ]
        weights = np.asarray(rarest, dtype=np.float64) ** -RARE_VERB_POWER
        return weights / weights.sum()

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
        *,
        seed: int,
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
        self._ids, self._cache, self.dropped_clip_ids = self.__load_windowable_clips(
            base_clip_ids(root, split)
        )
        log_dropped_clips(self.dropped_clip_ids, f"shorter than the window of {window} frames")
        if not self._ids:
            raise RuntimeError(f"no clips >= window {window} for split {split!r}")
        self._overflows = np.array([clip.shape[0] - window for clip in self._cache], np.int64)
        self._starts = self._overflows // 2
        self._starts_epoch = -1

    def __load_windowable_clips(
        self, names: list[str]
    ) -> tuple[list[str], list[np.ndarray], list[str]]:
        ids: list[str] = []
        short: list[str] = []
        clips: list[np.ndarray] = []
        non_finite: list[str] = []
        variants = [v for name in names for v in ([name, f"M{name}"] if self._mirror else [name])]
        check_no_missing_files(
            "feature", [v for v in variants if not (self._vec_dir / f"{v}.npy").is_file()]
        )
        for clip_id in variants:
            feat = np.load(self._vec_dir / f"{clip_id}.npy").astype(np.float32)
            if feat.shape[0] < self._window:
                short.append(clip_id)
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
        return ids, clips, short

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


def log_dropped_clips(dropped: list[str], reason: str) -> None:
    if dropped:
        logger.info(f"dropped {len(dropped)} clips {reason} (first: {dropped[:5]})")


def item_rng(seed: int, epoch: int, index: int) -> np.random.RandomState:
    item_seed = np.random.SeedSequence((seed, epoch, index)).generate_state(1)[0]
    return np.random.RandomState(item_seed)


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
        sampler = None if weights is None else self.__weighted_sampler(weights)
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

    def __weighted_sampler(self, weights: np.ndarray) -> WeightedRandomSampler:
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
