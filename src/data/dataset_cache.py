from __future__ import annotations

import gc
import hashlib
import logging
import os
from pathlib import Path

import joblib
import numpy as np

from src.data.amass import AMASSLoader
from src.data.augmentation import canonicalizeRoot, detectTpose, qualityFilter, resampleToFps
from src.data.motion_normalize import MotionStats, computeMotionStats

log = logging.getLogger(__name__)

CACHE_SCHEMA = 4  # bump when cache payload shape, filter params, or canonicalization changes

# Cache stores clips at this length cap (~33 s @ 30 fps), covering full content
# of >95% of AMASS clips. Runtime DataLoader random-crops to maxMotionLength (200).
# Larger = more temporal-window diversity per clip, but more disk + RAM.
INGEST_MAX_LENGTH = 1000

DEFAULT_FILTER_KWARGS: dict = {
    "minFrames": 30,
    "maxRootSpeed": 10.0,
    "maxAccel": 50.0,
    "maxJointRotvel": 30.0,
    "minVariance": 1e-4,
}


def processAmassFiles(
    loader: AMASSLoader, files: list, maxLength: int, filterKwargs: dict | None = None
) -> list[dict]:
    fkw = filterKwargs or DEFAULT_FILTER_KWARGS
    samples: list[dict] = []
    nLoaded = nSkipped = 0

    for i, fpath in enumerate(files):
        if i > 0 and i % 2000 == 0:
            log.info(
                "[MotionDataset] Processing %d / %d (kept=%d) ...",
                i, len(files), len(samples),
            )
        s = loader.loadFile(fpath)

        if s is None or s.motion.shape[0] < 4:
            nSkipped += 1
            continue

        motion, fps = s.motion, float(s.fps)
        trim_s, trim_e = detectTpose(motion)

        if trim_s > 0 or trim_e > 0:
            end = motion.shape[0] - trim_e if trim_e > 0 else motion.shape[0]
            motion = motion[trim_s:end]

        if abs(fps - 30.0) >= 0.5:
            motion = resampleToFps(motion, fps, 30.0)

        if not qualityFilter(motion, 30.0, **fkw):
            nSkipped += 1
            continue

        motion = canonicalizeRoot(motion[:maxLength])
        text = s.text if s.text else f"motion {s.sampleId.split('/')[-1]}"
        samples.append({"motion": motion.copy(), "text": text, "sample_id": s.sampleId})
        nLoaded += 1
        del s, motion

    gc.collect()
    log.info("AMASSLoader: loaded %d / %d sequences (skipped %d)", nLoaded, len(files), nSkipped)

    return samples


def loadOrBuildCache(
    dataDir: str, maxLength: int, maxSamples: int | None,
    filterKwargs: dict | None = None,
) -> tuple[list[dict], MotionStats]:
    fkw = filterKwargs or DEFAULT_FILTER_KWARGS
    fkwStr = ":".join(f"{k}={v}" for k, v in sorted(fkw.items()))
    ck = hashlib.md5(
        f"{dataDir}:{maxSamples}:{maxLength}:{fkwStr}:{CACHE_SCHEMA}".encode()
    ).hexdigest()[:12]
    cachePath = Path("data") / ".cache" / f"motion_dataset_{ck}.joblib"

    if cachePath.exists():
        log.info("[MotionDataset] Loading from cache: %s", cachePath)
        payload = joblib.load(cachePath)
        log.info("[MotionDataset] Cache hit: %d processed samples", len(payload["samples"]))

        return payload["samples"], payload["stats"]

    loader = AMASSLoader(dataDir)
    files = loader.discoverFiles()

    if maxSamples:
        files = files[:maxSamples]

    samples = processAmassFiles(loader, files, maxLength, fkw)
    stats = computeMotionStats(samples)
    os.makedirs(cachePath.parent, exist_ok=True)
    joblib.dump({"samples": samples, "stats": stats, "schema": CACHE_SCHEMA}, cachePath, compress=3)
    log.info("[MotionDataset] Cached %d samples (+ stats, mean-norm=%.4f) to %s",
             len(samples), float(np.abs(stats.mean).mean()), cachePath)

    return samples, stats

