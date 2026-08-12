from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE_NAME = "clip_lengths.json"


def _read_cache(cache_path: Path) -> dict[str, list[int]]:
    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in raw.items()}


def clip_lengths(vec_dir: Path, clip_ids: list[str]) -> dict[str, int]:
    cache_path = vec_dir.parent / CACHE_NAME
    cache = _read_cache(cache_path)

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
