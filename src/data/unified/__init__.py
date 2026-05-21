import hashlib
import logging
import os

import joblib

from src.data.augmentation import detect_tpose, quality_filter, resample_to_fps
from src.data.dataset_cache import INGEST_MAX_LENGTH, load_or_build_cache

from .unified_enrich import enrich_unified  # noqa: F401
from .unified_load import (  # noqa: F401
    load_amass,
    load_arctic,
    load_humanml3d,
    load_interx,
    preprocess_motion,
)
from .unified_split import split_samples  # noqa: F401

log = logging.getLogger(__name__)


def build_sources_buffer(cfg, min_frames: int, resample_fn, quality_fn, tpose_fn,
                       max_length: int | None = 200) -> list[dict]:
    """Load and enrich all enabled data sources into a flat buffer. Shared by dataset and trainer.

    max_length: if set, every motion is cropped to this many frames at ingest time.
    Default 200 caps memory; pass None to keep full clips (only safe for small subsets).
    """
    args = (min_frames, resample_fn, quality_fn, tpose_fn, max_length)
    buf: list[dict] = []

    if cfg.amass.enabled:
        load_amass(buf, cfg.amass, *args)

    if cfg.arctic.enabled:
        load_arctic(buf, cfg.arctic, *args)

    interx_cfg = getattr(cfg, "interx", None)

    if interx_cfg is not None and interx_cfg.enabled:
        load_interx(buf, interx_cfg, *args)

    humanml3d_cfg = getattr(cfg, "humanml3d", None)

    if humanml3d_cfg is not None and humanml3d_cfg.enabled:
        load_humanml3d(buf, humanml3d_cfg, *args)

    enrich_unified(buf)

    return buf


def build_or_load_unified_buffer(ucfg, src_key: str) -> list[dict]:
    """Memory-frugal cached unified buffer for training and eval.

    Reuses the existing AMASS motion_dataset cache (~676 MB on disk, pre-truncated)
    instead of re-streaming 16k raw .npz files into RAM, which OOMs on systems
    with other workloads competing for memory. HumanML3D portion is built fresh
    via build_sources_buffer with AMASS disabled (small, ~6k clips). The merged
    buffer is joblib-cached at data/.cache/eval_unified_buf_<hash>.joblib so
    repeat runs are instant.
    """
    key_parts = [src_key, str(INGEST_MAX_LENGTH),
                str(ucfg.amass.enabled), str(ucfg.arctic.enabled),
                str(getattr(ucfg, "humanml3d", None) and ucfg.humanml3d.enabled),
                str(getattr(ucfg, "interx", None) and ucfg.interx.enabled)]
    key_hash = hashlib.md5(":".join(key_parts).encode()).hexdigest()[:12]
    cache_path = os.path.join("data", ".cache", f"eval_unified_buf_{key_hash}.joblib")

    if os.path.exists(cache_path):
        log.info("[unified] loading buffer cache: %s", cache_path)
        buf = joblib.load(cache_path)
        log.info("[unified] cache hit: %d samples", len(buf))

        return buf

    log.info("[unified] building buffer (will cache to %s)", cache_path)
    buf: list[dict] = []
    loader_args = (30, resample_to_fps, quality_filter, detect_tpose, INGEST_MAX_LENGTH)

    if ucfg.amass.enabled:
        log.info("[unified]   loading AMASS via existing motion_dataset cache")
        amass_samples, _ = load_or_build_cache(ucfg.amass.data_dir, INGEST_MAX_LENGTH, None)

        for s in amass_samples:
            buf.append({**s, "source": "amass"})
        log.info("[unified]   AMASS: %d samples", len(amass_samples))

    if ucfg.arctic.enabled:
        log.info("[unified]   loading ARCTIC via fresh loader")
        arctic_buf: list[dict] = []
        load_arctic(arctic_buf, ucfg.arctic, *loader_args)
        buf.extend(arctic_buf)
        log.info("[unified]   ARCTIC: %d samples", len(arctic_buf))

    interx_cfg = getattr(ucfg, "interx", None)

    if interx_cfg is not None and interx_cfg.enabled:
        log.info("[unified]   loading InterX via fresh loader")
        ix_buf: list[dict] = []
        load_interx(ix_buf, interx_cfg, *loader_args)
        buf.extend(ix_buf)
        log.info("[unified]   InterX: %d samples", len(ix_buf))

    humanml3d_cfg = getattr(ucfg, "humanml3d", None)

    if humanml3d_cfg is not None and humanml3d_cfg.enabled:
        log.info("[unified]   loading HumanML3D fresh")
        human_buf: list[dict] = []
        load_humanml3d(human_buf, humanml3d_cfg, *loader_args)
        buf.extend(human_buf)
        log.info("[unified]   HumanML3D: %d samples", len(human_buf))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    joblib.dump(buf, cache_path, compress=3)
    log.info("[unified] cached %d total samples", len(buf))

    return buf
