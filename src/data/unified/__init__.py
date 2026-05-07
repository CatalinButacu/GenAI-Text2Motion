import hashlib
import logging
import os

import joblib

from src.data.augmentation import detectTpose, qualityFilter, resampleToFps
from src.data.dataset_cache import INGEST_MAX_LENGTH, loadOrBuildCache

from .unified_enrich import enrichUnified  # noqa: F401
from .unified_load import (  # noqa: F401
    loadAmass,
    loadArctic,
    loadHumanml3d,
    loadInterx,
    preprocessMotion,
)
from .unified_split import splitSamples  # noqa: F401

log = logging.getLogger(__name__)


def buildSourcesBuffer(cfg, minFrames: int, resampleFn, qualityFn, tposeFn,
                       maxLength: int | None = 200) -> list[dict]:
    """Load and enrich all enabled data sources into a flat buffer. Shared by dataset and trainer.

    maxLength: if set, every motion is cropped to this many frames at ingest time.
    Default 200 caps memory; pass None to keep full clips (only safe for small subsets).
    """
    args = (minFrames, resampleFn, qualityFn, tposeFn, maxLength)
    buf: list[dict] = []

    if cfg.amass.enabled:
        loadAmass(buf, cfg.amass, *args)

    if cfg.arctic.enabled:
        loadArctic(buf, cfg.arctic, *args)

    interxCfg = getattr(cfg, "interx", None)

    if interxCfg is not None and interxCfg.enabled:
        loadInterx(buf, interxCfg, *args)

    humanml3dCfg = getattr(cfg, "humanml3d", None)

    if humanml3dCfg is not None and humanml3dCfg.enabled:
        loadHumanml3d(buf, humanml3dCfg, *args)

    enrichUnified(buf)

    return buf


def buildOrLoadUnifiedBuffer(ucfg, srcKey: str) -> list[dict]:
    """Memory-frugal cached unified buffer for training and eval.

    Reuses the existing AMASS motion_dataset cache (~676 MB on disk, pre-truncated)
    instead of re-streaming 16k raw .npz files into RAM, which OOMs on systems
    with other workloads competing for memory. HumanML3D portion is built fresh
    via buildSourcesBuffer with AMASS disabled (small, ~6k clips). The merged
    buffer is joblib-cached at data/.cache/eval_unified_buf_<hash>.joblib so
    repeat runs are instant.
    """
    keyParts = [srcKey, str(INGEST_MAX_LENGTH),
                str(ucfg.amass.enabled), str(ucfg.arctic.enabled),
                str(getattr(ucfg, "humanml3d", None) and ucfg.humanml3d.enabled),
                str(getattr(ucfg, "interx", None) and ucfg.interx.enabled)]
    keyHash = hashlib.md5(":".join(keyParts).encode()).hexdigest()[:12]
    cachePath = os.path.join("data", ".cache", f"eval_unified_buf_{keyHash}.joblib")

    if os.path.exists(cachePath):
        log.info("[unified] loading buffer cache: %s", cachePath)
        buf = joblib.load(cachePath)
        log.info("[unified] cache hit: %d samples", len(buf))

        return buf

    log.info("[unified] building buffer (will cache to %s)", cachePath)
    buf: list[dict] = []
    loaderArgs = (30, resampleToFps, qualityFilter, detectTpose, INGEST_MAX_LENGTH)

    if ucfg.amass.enabled:
        log.info("[unified]   loading AMASS via existing motion_dataset cache")
        amassSamples, _ = loadOrBuildCache(ucfg.amass.dataDir, INGEST_MAX_LENGTH, None)

        for s in amassSamples:
            buf.append({**s, "source": "amass"})
        log.info("[unified]   AMASS: %d samples", len(amassSamples))

    if ucfg.arctic.enabled:
        log.info("[unified]   loading ARCTIC via fresh loader")
        arcticBuf: list[dict] = []
        loadArctic(arcticBuf, ucfg.arctic, *loaderArgs)
        buf.extend(arcticBuf)
        log.info("[unified]   ARCTIC: %d samples", len(arcticBuf))

    interxCfg = getattr(ucfg, "interx", None)

    if interxCfg is not None and interxCfg.enabled:
        log.info("[unified]   loading InterX via fresh loader")
        ixBuf: list[dict] = []
        loadInterx(ixBuf, interxCfg, *loaderArgs)
        buf.extend(ixBuf)
        log.info("[unified]   InterX: %d samples", len(ixBuf))

    humanml3dCfg = getattr(ucfg, "humanml3d", None)

    if humanml3dCfg is not None and humanml3dCfg.enabled:
        log.info("[unified]   loading HumanML3D fresh")
        humanBuf: list[dict] = []
        loadHumanml3d(humanBuf, humanml3dCfg, *loaderArgs)
        buf.extend(humanBuf)
        log.info("[unified]   HumanML3D: %d samples", len(humanBuf))

    os.makedirs(os.path.dirname(cachePath), exist_ok=True)
    joblib.dump(buf, cachePath, compress=3)
    log.info("[unified] cached %d total samples", len(buf))

    return buf
