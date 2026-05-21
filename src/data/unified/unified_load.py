from __future__ import annotations

import logging
from pathlib import Path

from src.data.amass import AMASSLoader, ARCTICLoader, InterXLoader
from src.data.augmentation import canonicalize_root
from src.data.humanml3d import HumanML3DLoader, build_norm_map, parse_index_csv, preload_smplx

log = logging.getLogger(__name__)


def preprocess_motion(motion, fps: float, resample_fn, qfilt_fn, detect_fn,
                     max_length: int | None = None):
    trim_s, trim_e = detect_fn(motion)

    if trim_s > 0 or trim_e > 0:
        end = motion.shape[0] - trim_e if trim_e > 0 else motion.shape[0]
        motion = motion[trim_s:end]

    if abs(fps - 30.0) >= 0.5:
        motion = resample_fn(motion, fps, 30.0)

    if not qfilt_fn(motion, 30.0):
        return None

    if max_length is not None and motion.shape[0] > max_length:
        motion = motion[:max_length]

    return motion


def load_amass(buf: list, cfg, min_frames: int, resample_fn, qfilt_fn, detect_fn,
              max_length: int | None = None) -> None:
    loader = AMASSLoader(cfg.data_dir)
    samples = loader.load_dataset(max_samples=cfg.max_samples, min_frames=4)
    n = 0

    for s in samples:
        motion = preprocess_motion(s.motion, s.fps, resample_fn, qfilt_fn, detect_fn, max_length)

        if motion is None or motion.shape[0] < min_frames:
            continue

        text = s.text or f"motion {s.sample_id.split('/')[-1]}"
        buf.append({"motion": motion, "text": text, "source": "amass", "sample_id": s.sample_id})
        n += 1
    log.info("[UnifiedDataset] AMASS: %d / %d", n, len(samples))


def load_arctic(buf: list, cfg, min_frames: int, resample_fn, qfilt_fn, detect_fn,
               max_length: int | None = None) -> None:
    loader = ARCTICLoader(cfg.data_dir)
    samples = loader.load_dataset(max_samples=cfg.max_samples, min_frames=4)
    n = 0

    for s in samples:
        motion = preprocess_motion(s.motion, s.fps, resample_fn, qfilt_fn, detect_fn, max_length)

        if motion is None or motion.shape[0] < min_frames:
            continue

        text = s.text or f"person interacts with object {s.sample_id}"
        buf.append({"motion": motion, "text": text, "source": "arctic", "sample_id": s.sample_id})
        n += 1
    log.info("[UnifiedDataset] ARCTIC: %d / %d", n, len(samples))


def load_interx(buf: list, cfg, min_frames: int, resample_fn, qfilt_fn, detect_fn,
               max_length: int | None = None) -> None:
    loader = InterXLoader(cfg.data_dir)
    samples = loader.load_dataset(max_samples=cfg.max_samples, min_frames=4)
    n = 0

    for s in samples:
        motion = preprocess_motion(s.motion, s.fps, resample_fn, qfilt_fn, detect_fn, max_length)

        if motion is None or motion.shape[0] < min_frames:
            continue

        text = s.text or f"two-person interaction {s.sample_id}"
        # interaction_id groups P1+P2 of the same Inter-X sequence into one split bucket
        # (sample_id = "interx/<seq_id>/P{1|2}") -- prevents two-agent test-set leakage.
        seq_id = s.sample_id.split("/")[1] if "/" in s.sample_id else s.sample_id
        buf.append({"motion": motion, "text": text, "source": "interx",
                    "sample_id": s.sample_id, "interaction_id": f"interx:{seq_id}"})
        n += 1
    log.info("[UnifiedDataset] InterX: %d / %d", n, len(samples))


def load_humanml3d(buf: list, cfg, min_frames: int, resample_fn, qfilt_fn, detect_fn,
                  max_length: int | None = None) -> None:
    """Load HumanML3D text-motion pairs into the unified buffer.

    Requires data/humanml3d/{texts,split,index.csv} and AMASS .npz files.
    cfg.data_dir  -- HumanML3D root (texts/, split/, index.csv)
    cfg.amass_dir -- AMASS root for the backing motion clips
    """
    hml_dir = Path(cfg.data_dir)
    amass_dir = Path(cfg.amass_dir)
    idx_f = hml_dir / "index.csv"

    if not idx_f.exists():
        log.warning("[UnifiedDataset] HumanML3D: index.csv not found at %s -- skipping", idx_f)
        return

    loader = HumanML3DLoader(str(hml_dir))
    local_amass = AMASSLoader(str(amass_dir))
    norm_map = build_norm_map(local_amass.discover_files(), amass_dir)
    all_texts = {item["clip_id"]: item["texts"] for item in loader.load_texts()}

    all_split_ids: set[str] = set()
    for split in ("train", "val", "test"):
        try:
            all_split_ids.update(loader.load_split(split))
        except FileNotFoundError:
            log.debug("[unified] HumanML3D split file missing: %s -- skipping", split)

    samples = parse_index_csv(idx_f, all_split_ids, all_texts, norm_map)
    if cfg.max_samples:
        samples = samples[:cfg.max_samples]

    cache, bad = preload_smplx(samples, amass_dir)
    if bad:
        bad_set = set(bad)
        samples = [s for s in samples if s["clip_id"] not in bad_set]

    n = 0
    n_rejected = 0
    for s in samples:
        clip_id = s["clip_id"]
        motion = cache.get(clip_id)

        if motion is None or motion.shape[0] < min_frames:
            n_rejected += 1
            continue

        if not qfilt_fn(motion, 30.0):
            n_rejected += 1
            continue
        motion = canonicalize_root(motion)

        if max_length is not None and motion.shape[0] > max_length:
            motion = motion[:max_length]
        text = s.get("text") or (s["texts"][0] if s.get("texts") else "")
        buf.append({"motion": motion.copy(), "text": text, "source": "humanml3d",
                    "sample_id": clip_id})
        n += 1
    log.info("[UnifiedDataset] HumanML3D: %d kept / %d rejected (of %d)",
             n, n_rejected, len(samples))
