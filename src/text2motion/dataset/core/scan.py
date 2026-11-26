from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_DIM = 263
COLUMNS = ("clip", "frames", "finite", "all_zero", "shape_ok", "long_enough", "action")


def problems_of(array: np.ndarray) -> list[str]:
    found: list[str] = []
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        found.append("wrong shape")
    if not np.isfinite(array).all():
        found.append("non-finite values")
    if array.size > 0 and np.all(array == 0):
        found.append("all zero")
    return found


def check_clip(path: Path, min_frames: int) -> dict[str, object]:
    array = np.load(path)
    problems = problems_of(array)
    frames = int(array.shape[0]) if array.ndim >= 1 else 0
    long_enough = frames >= min_frames
    if problems:
        action = "exclude: " + ", ".join(problems)
    elif not long_enough:
        action = "filter by min length"
    else:
        action = "keep"
    return {
        "clip": path.stem,
        "frames": frames,
        "finite": bool(np.isfinite(array).all()),
        "all_zero": bool(array.size > 0 and np.all(array == 0)),
        "shape_ok": array.ndim == 2 and array.shape[1] == FEATURE_DIM,
        "long_enough": long_enough,
        "action": action,
    }


def scan_folder(folder: Path, log: Path, min_frames: int) -> dict[str, int]:
    rows: list[dict[str, Any]] = [
        check_clip(path, min_frames) for path in sorted(folder.glob("*.npy"))
    ]
    if not rows:
        raise FileNotFoundError(f"no .npy files in {folder}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = list(COLUMNS)
        writer: csv.DictWriter[str] = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["action"])] = counts.get(str(row["action"]), 0) + 1
    return counts
