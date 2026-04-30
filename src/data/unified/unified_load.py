from __future__ import annotations

import logging
from pathlib import Path

from src.data.amass import AMASSLoader, ARCTICLoader
from src.data.augmentation import canonicalizeRoot
from src.data.humanml3d import HumanML3DLoader, buildNormMap, parseIndexCsv, preloadSmplx

log = logging.getLogger(__name__)


def preprocessMotion(motion, fps: float, resampleFn, qfiltFn, detectFn,
                     maxLength: int | None = None):
    trim_s, trim_e = detectFn(motion)

    if trim_s > 0 or trim_e > 0:
        end = motion.shape[0] - trim_e if trim_e > 0 else motion.shape[0]
        motion = motion[trim_s:end]

    if abs(fps - 30.0) >= 0.5:
        motion = resampleFn(motion, fps, 30.0)

    if not qfiltFn(motion, 30.0):
        return None

    if maxLength is not None and motion.shape[0] > maxLength:
        motion = motion[:maxLength]

    return motion


def loadAmass(buf: list, cfg, minFrames: int, resampleFn, qfiltFn, detectFn,
              maxLength: int | None = None) -> None:
    loader = AMASSLoader(cfg.dataDir)
    samples = loader.loadDataset(maxSamples=cfg.maxSamples, minFrames=4)
    n = 0

    for s in samples:
        motion = preprocessMotion(s.motion, s.fps, resampleFn, qfiltFn, detectFn, maxLength)

        if motion is None or motion.shape[0] < minFrames:
            continue

        text = s.text or f"motion {s.sampleId.split('/')[-1]}"
        buf.append({"motion": motion, "text": text, "source": "amass", "sample_id": s.sampleId})
        n += 1
    log.info("[UnifiedDataset] AMASS: %d / %d", n, len(samples))


def loadArctic(buf: list, cfg, minFrames: int, resampleFn, qfiltFn, detectFn,
               maxLength: int | None = None) -> None:
    loader = ARCTICLoader(cfg.dataDir)
    samples = loader.loadDataset(maxSamples=cfg.maxSamples, minFrames=4)
    n = 0

    for s in samples:
        motion = preprocessMotion(s.motion, s.fps, resampleFn, qfiltFn, detectFn, maxLength)

        if motion is None or motion.shape[0] < minFrames:
            continue

        text = s.text or f"person interacts with object {s.sampleId}"
        buf.append({"motion": motion, "text": text, "source": "arctic"})
        n += 1
    log.info("[UnifiedDataset] ARCTIC: %d / %d", n, len(samples))


def loadHumanml3d(buf: list, cfg, minFrames: int, resampleFn, qfiltFn, detectFn,
                  maxLength: int | None = None) -> None:
    """Load HumanML3D text-motion pairs into the unified buffer.

    Requires data/humanml3d/{texts,split,index.csv} and AMASS .npz files.
    cfg.data_dir  -- HumanML3D root (texts/, split/, index.csv)
    cfg.amass_dir -- AMASS root for the backing motion clips
    """
    hmlDir = Path(cfg.dataDir)
    amassDir = Path(cfg.amassDir)
    idxF = hmlDir / "index.csv"

    if not idxF.exists():
        log.warning("[UnifiedDataset] HumanML3D: index.csv not found at %s -- skipping", idxF)
        return

    loader = HumanML3DLoader(str(hmlDir))
    localAmass = AMASSLoader(str(amassDir))
    normMap = buildNormMap(localAmass.discoverFiles(), amassDir)
    allTexts = {item["clip_id"]: item["texts"] for item in loader.loadTexts()}

    allSplitIds: set[str] = set()
    for split in ("train", "val", "test"):
        try:
            allSplitIds.update(loader.loadSplit(split))
        except FileNotFoundError:
            log.debug("[unified] HumanML3D split file missing: %s -- skipping", split)

    samples = parseIndexCsv(idxF, allSplitIds, allTexts, normMap)
    if cfg.maxSamples:
        samples = samples[:cfg.maxSamples]

    cache, bad = preloadSmplx(samples, amassDir)
    if bad:
        badSet = set(bad)
        samples = [s for s in samples if s["clip_id"] not in badSet]

    n = 0
    nRejected = 0
    for s in samples:
        clipId = s["clip_id"]
        motion = cache.get(clipId)

        if motion is None or motion.shape[0] < minFrames:
            nRejected += 1
            continue

        if not qfiltFn(motion, 30.0):
            nRejected += 1
            continue
        motion = canonicalizeRoot(motion)

        if maxLength is not None and motion.shape[0] > maxLength:
            motion = motion[:maxLength]
        text = s.get("text") or (s["texts"][0] if s.get("texts") else "")
        buf.append({"motion": motion.copy(), "text": text, "source": "humanml3d",
                    "sample_id": clipId})
        n += 1
    log.info("[UnifiedDataset] HumanML3D: %d kept / %d rejected (of %d)",
             n, nRejected, len(samples))
