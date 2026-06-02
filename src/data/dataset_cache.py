from __future__ import annotations

import gc
import hashlib
import logging
import os
from pathlib import Path

import joblib
import numpy as np

from src.data.amass import AMASSLoader
from src.data.augmentation import canonicalize_root, detect_tpose, quality_filter, resample_to_fps
from src.data.motion_normalize import MotionStats, compute_motion_stats
from src.shared.constants import CACHE_SCHEMA, DEFAULT_FILTER_KWARGS, INGEST_MAX_LENGTH

log = logging.getLogger(__name__)


def process_amass_files(
    loader: AMASSLoader, files: list, max_length: int, filter_kwargs: dict | None = None
) -> list[dict]:
    fkw = filter_kwargs or DEFAULT_FILTER_KWARGS
    samples: list[dict] = []
    n_loaded = n_skipped = 0

    for i, fpath in enumerate(files):
        if i > 0 and i % 2000 == 0:
            log.info(
                "[MotionDataset] Processing %d / %d (kept=%d) ...",
                i, len(files), len(samples),
            )
        s = loader.load_file(fpath)

        if s is None or s.motion.shape[0] < 4:
            n_skipped += 1
            continue

        motion, fps = s.motion, float(s.fps)
        trim_s, trim_e = detect_tpose(motion)

        if trim_s > 0 or trim_e > 0:
            end = motion.shape[0] - trim_e if trim_e > 0 else motion.shape[0]
            motion = motion[trim_s:end]

        if abs(fps - 30.0) >= 0.5:
            motion = resample_to_fps(motion, fps, 30.0)

        if not quality_filter(motion, 30.0, **fkw):
            n_skipped += 1
            continue

        motion = canonicalize_root(motion[:max_length])
        text = s.text if s.text else f"motion {s.sample_id.split('/')[-1]}"
        samples.append({"motion": motion.copy(), "text": text, "sample_id": s.sample_id})
        n_loaded += 1
        del s, motion

    gc.collect()
    log.info("AMASSLoader: loaded %d / %d sequences (skipped %d)", n_loaded, len(files), n_skipped)

    return samples


def load_or_build_cache(
    data_dir: str, max_length: int, max_samples: int | None,
    filter_kwargs: dict | None = None,
) -> tuple[list[dict], MotionStats]:
    fkw = filter_kwargs or DEFAULT_FILTER_KWARGS
    fkw_str = ":".join(f"{k}={v}" for k, v in sorted(fkw.items()))
    ck = hashlib.md5(
        f"{data_dir}:{max_samples}:{max_length}:{fkw_str}:{CACHE_SCHEMA}".encode()
    ).hexdigest()[:12]
    cache_path = Path("data") / ".cache" / f"motion_dataset_{ck}.joblib"

    if cache_path.exists():
        log.info("[MotionDataset] Loading from cache: %s", cache_path)
        payload = joblib.load(cache_path)
        log.info("[MotionDataset] Cache hit: %d processed samples", len(payload["samples"]))

        return payload["samples"], payload["stats"]

    loader = AMASSLoader(data_dir)
    files = loader.discover_files()

    if max_samples:
        files = files[:max_samples]

    samples = process_amass_files(loader, files, max_length, fkw)
    stats = compute_motion_stats(samples)
    os.makedirs(cache_path.parent, exist_ok=True)
    joblib.dump(
        {"samples": samples, "stats": stats, "schema": CACHE_SCHEMA},
        cache_path, compress=3,
    )
    log.info("[MotionDataset] Cached %d samples (+ stats, mean-norm=%.4f) to %s",
             len(samples), float(np.abs(stats.mean).mean()), cache_path)

    return samples, stats

