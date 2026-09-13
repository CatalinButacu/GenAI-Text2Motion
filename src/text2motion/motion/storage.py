from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from text2motion.motion.contracts import Split

FEATURES_DIR = "new_joint_vecs"
TEXTS_DIR = "texts"
MEAN_FILE = "Mean.npy"
STD_FILE = "Std.npy"
STATS_PROVENANCE_FILE = "stats_provenance.json"
CLIP_LENGTHS_FILE = "clip_lengths.json"


SPLIT_FILES = {split: f"{split.value}.txt" for split in Split}


def item_rng(seed: int, epoch: int, index: int) -> np.random.RandomState:
    item_seed = np.random.SeedSequence((seed, epoch, index)).generate_state(1)[0]
    return np.random.RandomState(item_seed)


def _read_frame_count_cache(cache_path: Path) -> dict[str, list[int]]:
    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {key: list(value) for key, value in raw.items()}


def clip_frame_counts(vec_dir: Path, clip_ids: list[str]) -> dict[str, int]:
    cache_path = vec_dir.parent / CLIP_LENGTHS_FILE
    cache = _read_frame_count_cache(cache_path)

    lengths: dict[str, int] = {}
    dirty = False
    for clip_id in clip_ids:
        path = vec_dir / f"{clip_id}.npy"
        if not path.is_file():
            continue
        size = path.stat().st_size
        entry = cache.get(clip_id)
        if entry is not None and entry[1] == size:
            lengths[clip_id] = entry[0]
            continue
        length = int(np.load(path, mmap_mode="r").shape[0])
        lengths[clip_id] = length
        cache[clip_id] = [length, size]
        dirty = True

    if dirty:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return lengths
