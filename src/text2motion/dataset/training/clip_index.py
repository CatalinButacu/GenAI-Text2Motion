from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.dataset.splits.train_access import split_clip_ids
from text2motion.dataset.training.batching import log_dropped_clips
from text2motion.dataset.training.captions import usable_annotations
from text2motion.motion.contracts import Split
from text2motion.motion.representation import DIM
from text2motion.motion.storage import (
    FEATURES_DIR,
    TEXTS_DIR,
    parse_text_file,
)

CLIP_LENGTHS_FILE = "clip_lengths.json"


def read_frame_count_cache(cache_path: Path) -> dict[str, list[int]]:
    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {key: list(value) for key, value in raw.items()}


def write_frame_count_cache(cache_path: Path, cache: dict[str, list[int]]) -> None:
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(cache), encoding="utf-8")
    temporary.replace(cache_path)


def clip_frame_counts(vec_dir: Path, clip_ids: list[str]) -> dict[str, int]:
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
        write_frame_count_cache(cache_path, cache)
    return lengths


def check_no_missing_files(kind: str, clip_ids: list[str]) -> None:
    if clip_ids:
        raise FileNotFoundError(
            f"{len(clip_ids)} clips of the split list have no {kind} file; first: {clip_ids[:5]}"
        )


############################################################
#       ClipIndex
############################################################
class ClipIndex:
    def __init__(
        self,
        root: Path,
        data: MotionDataConfig,
        split: Split,
        texts_dir: Path | None = None,
    ) -> None:
        self._min_len = data.min_motion_len
        self._mirror = data.mirror_augment and split == Split.TRAIN
        self._vec_dir = root / FEATURES_DIR
        self._text_dir = Path(texts_dir) if texts_dir is not None else root / TEXTS_DIR

        self.ids, self.dropped_clip_ids = self.select_usable_clip_ids(split_clip_ids(root, split))
        log_dropped_clips(self.dropped_clip_ids, f"shorter than {self._min_len} frames")
        if not self.ids:
            raise RuntimeError(f"no usable clips for split {split!r} under {self._vec_dir}")
        self.annotations = {
            clip_id: parse_text_file(self._text_dir / f"{clip_id}.txt") for clip_id in self.ids
        }
        self.check_captions_cover_minimum_length(split)

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

    def frame_counts(self) -> dict[str, int]:
        return clip_frame_counts(self._vec_dir, self.ids)

    def check_captions_cover_minimum_length(self, split: Split) -> None:
        lengths = self.frame_counts()
        uncaptioned = [
            clip_id
            for clip_id in self.ids
            if not usable_annotations(self.annotations[clip_id], lengths[clip_id], self._min_len)
        ]
        if uncaptioned:
            raise ValueError(
                f"{len(uncaptioned)} clips of split {split!r} have no caption that covers "
                f"{self._min_len} frames (first: {uncaptioned[:5]}); remove them from the split list"
            )

    def load_features(self, clip_id: str) -> np.ndarray:
        feat = np.load(self._vec_dir / f"{clip_id}.npy").astype(np.float32)
        if feat.shape[-1] != DIM:
            raise ValueError(f"{clip_id}: expected {DIM}-dim feature, got {feat.shape}")
        if not np.isfinite(feat).all():
            raise ValueError(
                f"{clip_id}: non-finite values in feature; drop it from the split list"
            )
        return feat
