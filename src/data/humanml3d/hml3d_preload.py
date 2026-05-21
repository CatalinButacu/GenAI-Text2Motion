from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.data.amass import AMASSLoader
from src.data.augmentation import canonicalize_root, resample_to_fps

log = logging.getLogger(__name__)


def slice_amass_motion(sample: dict, amass: AMASSLoader) -> np.ndarray:
    seq = amass.load_file(Path(sample["amass_path"]))

    if seq is None:
        raise ValueError("load failed")

    motion = seq.motion
    start = max(0, min(sample["start"], motion.shape[0] - 1))
    end = min(sample["end"] if sample["end"] != -1 else motion.shape[0], motion.shape[0])
    end = max(start + 1, end)
    sliced = motion[start:end]

    if abs(seq.fps - 30.0) >= 0.5:
        sliced = resample_to_fps(sliced, seq.fps, 30.0)

    return sliced.astype(np.float32)


def preload_smplx(samples: list[dict], amass_dir: Path) -> tuple[dict, list]:
    amass = AMASSLoader(str(amass_dir))
    cache: dict[str, np.ndarray] = {}
    bad: list[str] = []

    for s in samples:
        try:
            cache[s["clip_id"]] = slice_amass_motion(s, amass)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("[HumanML3D] skipping %s: %s", s["amass_path"], exc)
            bad.append(s["clip_id"])

    return cache, bad


def load_motion_item(s: dict, cache: dict | None, amass_dir: Path) -> np.ndarray:
    if cache is not None:
        return cache[s["clip_id"]].copy()

    return canonicalize_root(slice_amass_motion(s, AMASSLoader(str(amass_dir))))
