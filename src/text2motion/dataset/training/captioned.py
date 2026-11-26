from __future__ import annotations

import json
import logging
import os
from collections import Counter
from multiprocessing import RawValue
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from text2motion.dataset.contracts import CaptionWeighting, ClipWeighting, MotionDataConfig
from text2motion.dataset.splits.view import split_clip_ids
from text2motion.dataset.training.batching import item_rng, log_dropped_clips
from text2motion.dataset.training.labels import (
    MOTION_CLASS_POWER,
    NO_MOTION_CLASS,
    NO_VERB_GROUP,
    RARE_VERB_MIN_CLIPS,
    RARE_VERB_POWER,
    VERB_BALANCE_MIN_CLIPS,
    caption_verbs,
    read_motion_classes,
)
from text2motion.motion.contracts import (
    MotionClip,
    Split,
    TextAnnotation,
)
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM, FPS
from text2motion.motion.storage import (
    FEATURES_DIR,
    TEXTS_DIR,
    parse_text_file,
)

logger = logging.getLogger(__name__)


CLIP_LENGTHS_FILE = "clip_lengths.json"


def clip_frame_counts(vec_dir: Path, clip_ids: list[str]) -> dict[str, int]:
    def read_frame_count_cache(cache_path: Path) -> dict[str, list[int]]:
        if not cache_path.is_file():
            return {}
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {key: list(value) for key, value in raw.items()}

    cache_path = vec_dir.parent / CLIP_LENGTHS_FILE
    cache = read_frame_count_cache(cache_path)

    lengths: dict[str, int] = {}
    dirty = False
    for clip_id in clip_ids:
        path = vec_dir / f"{clip_id}.npy"
        if not path.is_file():
            continue
        stat = path.stat()
        identity = [stat.st_size, stat.st_mtime_ns]
        entry = cache.get(clip_id)
        if entry is not None and entry[1:] == identity:
            lengths[clip_id] = entry[0]
            continue
        length = int(np.load(path, mmap_mode="r").shape[0])
        lengths[clip_id] = length
        cache[clip_id] = [length, *identity]
        dirty = True

    if dirty:
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(cache), encoding="utf-8")
        temporary.replace(cache_path)
    return lengths


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

        self._ids, self.dropped_clip_ids = self.select_usable_clip_ids(split_clip_ids(root, split))
        log_dropped_clips(self.dropped_clip_ids, f"shorter than {self._min_len} frames")
        if not self._ids:
            raise RuntimeError(f"no usable clips for split {split!r} under {self._vec_dir}")
        self._annotations = {
            clip_id: parse_text_file(self._text_dir / f"{clip_id}.txt") for clip_id in self._ids
        }
        uncaptioned = self.clips_without_usable_caption()
        if uncaptioned:
            raise ValueError(
                f"{len(uncaptioned)} clips of split {split!r} have no caption that covers "
                f"{self._min_len} frames (first: {uncaptioned[:5]}); remove them from the split list"
            )
        self._verb_clip_counts: dict[str, int] | None = None
        if data.caption_weighting == CaptionWeighting.RARE_VERB and self._train:
            self._verb_clip_counts = self.count_verb_clips(RARE_VERB_MIN_CLIPS, "rare_verb caption")
        self._verb_group_sizes: dict[str, int] | None = None
        self._clip_weights: np.ndarray | None = None
        if data.clip_weighting == ClipWeighting.MOTION_CLASS and self._train:
            if data.motion_classes is None:
                raise ValueError("motion_class clip weighting needs the motion_classes verb map")
            clip_classes = self.clip_classes(read_motion_classes(data.motion_classes))
            self._clip_weights = self.motion_class_weights(clip_classes)
        if data.clip_weighting == ClipWeighting.UNIFORM_REPLACEMENT and self._train:
            self._clip_weights = np.ones(len(self._ids), dtype=np.float64)
        if data.clip_weighting == ClipWeighting.VERB and self._train:
            balance_counts = self.count_verb_clips(VERB_BALANCE_MIN_CLIPS, "verb clip")
            self._verb_clip_counts = balance_counts
            caption_groups = self.caption_groups(balance_counts)
            self._verb_group_sizes = self.verb_group_sizes(caption_groups)
            self._clip_weights = self.verb_group_weights(caption_groups, self._verb_group_sizes)

    def clip_verbs(self) -> list[frozenset[str]]:
        found = []
        for clip_id in self._ids:
            annotations = self._annotations[clip_id]
            found.append(frozenset().union(*(caption_verbs(a) for a in annotations)))
        return found

    def clip_classes(self, classes: dict[str, int]) -> list[frozenset[int]]:
        found = []
        for verbs in self.clip_verbs():
            labels = frozenset(classes[verb] for verb in verbs if verb in classes)
            found.append(labels or frozenset({NO_MOTION_CLASS}))
        return found

    @staticmethod
    def rarest_verb(annotation: TextAnnotation, counts: dict[str, int]) -> str:
        kept = [verb for verb in caption_verbs(annotation) if verb in counts]
        return min(kept, key=lambda verb: (counts[verb], verb), default=NO_VERB_GROUP)

    def caption_groups(self, counts: dict[str, int]) -> list[list[str]]:
        lengths = clip_frame_counts(self._vec_dir, self._ids)
        groups = []
        for clip_id in self._ids:
            annotations = self._annotations[clip_id]
            usable = self.usable_annotations(annotations, lengths[clip_id])
            groups.append([self.rarest_verb(annotation, counts) for annotation, _ in usable])
        return groups

    @staticmethod
    def verb_group_sizes(caption_groups: list[list[str]]) -> dict[str, int]:
        sizes = Counter(group for groups in caption_groups for group in groups)
        verb_sizes = [size for group, size in sizes.items() if group != NO_VERB_GROUP]
        sizes[NO_VERB_GROUP] = max(verb_sizes)
        return dict(sizes)

    @staticmethod
    def verb_group_weights(caption_groups: list[list[str]], sizes: dict[str, int]) -> np.ndarray:
        weights = [sum(1.0 / sizes[group] for group in groups) for groups in caption_groups]
        return np.asarray(weights, dtype=np.float64)

    @staticmethod
    def motion_class_weights(clip_classes: list[frozenset[int]]) -> np.ndarray:
        sizes = Counter(label for labels in clip_classes for label in labels)
        labelled = [size for label, size in sizes.items() if label != NO_MOTION_CLASS]
        if not labelled:
            raise ValueError("no train clip has a verb of the motion class map")
        sizes[NO_MOTION_CLASS] = max(labelled)
        rarest = [min(sizes[label] for label in labels) for labels in clip_classes]
        return np.asarray(rarest, dtype=np.float64) ** -MOTION_CLASS_POWER

    def count_verb_clips(self, min_clips: int, weighting: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for clip_id in self._ids:
            if clip_id.startswith("M"):
                continue
            annotations = self._annotations[clip_id]
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

    def captions(self) -> list[str]:
        return sorted(
            {
                annotation.caption
                for annotations in self._annotations.values()
                for annotation in annotations
            }
        )

    @property
    def mean(self) -> np.ndarray:
        return self.scaler.mean

    @property
    def std(self) -> np.ndarray:
        return self.scaler.std

    def clip_id_variants(self, name: str) -> list[str]:
        return [name, f"M{name}"] if self._mirror else [name]

    def select_usable_clip_ids(self, names: list[str]) -> tuple[list[str], list[str]]:
        candidates = [clip_id for name in names for clip_id in self.clip_id_variants(name)]
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
        annotations = self._annotations[clip_id]
        annotation, feat = self.select_captioned_segment(annotations, feat, rng)
        feat = self.normalize(self.crop_to_max_frames(feat, rng))
        return MotionClip(
            features=torch.from_numpy(feat),
            frame_count=feat.shape[0],
            caption=annotation.caption,
            clip_id=clip_id,
        )

    @staticmethod
    def annotation_frame_range(annotation: TextAnnotation, total: int) -> tuple[int, int] | None:
        if annotation.start_time == 0.0 and annotation.end_time == 0.0:
            return None
        start = max(int(annotation.start_time * FPS), 0)
        end = min(int(annotation.end_time * FPS), total)
        return start, end

    def usable_annotations(
        self, annotations: list[TextAnnotation], total: int
    ) -> list[tuple[TextAnnotation, tuple[int, int] | None]]:
        usable = []
        for annotation in annotations:
            segment = self.annotation_frame_range(annotation, total)
            if segment is None or segment[1] - segment[0] >= self._min_len:
                usable.append((annotation, segment))
        return usable

    def clips_without_usable_caption(self) -> list[str]:
        lengths = clip_frame_counts(self._vec_dir, self._ids)
        return [
            clip_id
            for clip_id in self._ids
            if not self.usable_annotations(self._annotations[clip_id], lengths[clip_id])
        ]

    def select_captioned_segment(
        self, annotations: list[TextAnnotation], feat: np.ndarray, rng: np.random.RandomState
    ) -> tuple[TextAnnotation, np.ndarray]:
        usable = self.usable_annotations(annotations, feat.shape[0])
        if not usable:
            raise ValueError("no caption of this clip covers a segment of the minimum length")
        annotation, segment = usable[self.caption_index([a for a, _ in usable], rng)]
        return annotation, feat if segment is None else feat[segment[0] : segment[1]]

    def caption_index(self, annotations: list[TextAnnotation], rng: np.random.RandomState) -> int:
        if self._verb_clip_counts is None:
            return rng.randint(len(annotations))
        return int(rng.choice(len(annotations), p=self.caption_weights(annotations)))

    def caption_weights(self, annotations: list[TextAnnotation]) -> np.ndarray:
        counts = self._verb_clip_counts
        if counts is None:
            return np.full(len(annotations), 1.0 / len(annotations))
        sizes = self._verb_group_sizes
        if sizes is not None:
            groups = [self.rarest_verb(annotation, counts) for annotation in annotations]
            weights = np.asarray([1.0 / sizes[group] for group in groups], dtype=np.float64)
            return weights / weights.sum()
        common = max(counts.values())
        rarest = [
            min((counts[verb] for verb in caption_verbs(a) if verb in counts), default=common)
            for a in annotations
        ]
        weights = np.asarray(rarest, dtype=np.float64) ** -RARE_VERB_POWER
        return weights / weights.sum()

    def crop_to_max_frames(self, feat: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        overflow = feat.shape[0] - self._max_len
        if overflow <= 0:
            return feat
        start = int(rng.randint(overflow + 1)) if self._train else 0
        return feat[start : start + self._max_len]
