from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.data.amass import AMASSLoader
from src.data.augmentation import canonicalizeRoot, resampleToFps

log = logging.getLogger(__name__)


def sliceAmassMotion(sample: dict, amass: AMASSLoader) -> np.ndarray:
    seq = amass.loadFile(Path(sample["amass_path"]))

    if seq is None:
        raise ValueError("load failed")

    motion = seq.motion
    start = max(0, min(sample["start"], motion.shape[0] - 1))
    end = min(sample["end"] if sample["end"] != -1 else motion.shape[0], motion.shape[0])
    end = max(start + 1, end)
    sliced = motion[start:end]

    if abs(seq.fps - 30.0) >= 0.5:
        sliced = resampleToFps(sliced, seq.fps, 30.0)

    return sliced.astype(np.float32)


def preloadSmplx(samples: list[dict], amassDir: Path) -> tuple[dict, list]:
    amass = AMASSLoader(str(amassDir))
    cache: dict[str, np.ndarray] = {}
    bad: list[str] = []

    for s in samples:
        try:
            cache[s["clip_id"]] = sliceAmassMotion(s, amass)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("[HumanML3D] skipping %s: %s", s["amass_path"], exc)
            bad.append(s["clip_id"])

    return cache, bad


def loadMotionItem(s: dict, cache: dict | None, amassDir: Path) -> np.ndarray:
    if cache is not None:
        return cache[s["clip_id"]].copy()

    return canonicalizeRoot(sliceAmassMotion(s, AMASSLoader(str(amassDir))))
