#!/usr/bin/env python
"""Per-criterion data-quality rejection stats.

Walks the configured AMASS and/or HumanML3D corpus, applies the same ingestion
pipeline used by the trainers (load -> detect_tpose -> resample_to_30fps ->
quality_filter), and reports how many clips failed each individual criterion.

Unlike the production quality_filter (returns a single bool), this script runs
every criterion *independently* on each clip so we can attribute rejections.
A clip can fail more than one criterion simultaneously; counts therefore sum
to >= total_rejected.

Usage
-----
    # AMASS only (~16k clips, ~3 min on warm cache)
    python scripts/data/quality_stats.py --source amass

    # HumanML3D only (uses index.csv mapping onto AMASS .npz files)
    python scripts/data/quality_stats.py --source humanml3d

    # Both, with default thresholds
    python scripts/data/quality_stats.py --source all

    # Override a threshold and write JSON
    python scripts/data/quality_stats.py --source amass --max-accel 80 \\
        --output results/quality_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.amass import AMASSLoader
from src.data.aug_quality import geodesic_joint_vel
from src.data.augmentation import detect_tpose, resample_to_fps
from src.data.dataset_cache import DEFAULT_FILTER_KWARGS
from src.data.humanml3d import (
    DEFAULT_DIR as HML3D_DIR,
)
from src.data.humanml3d import (
    HumanML3DLoader,
    build_norm_map,
    parse_index_csv,
    preload_smplx,
)

log = logging.getLogger(__name__)

REASONS: tuple[str, ...] = (
    "tooShort",        # T < min_frames
    "nanInf",          # any NaN/Inf in the pose tensor
    "fpsBad",          # native fps <= 1 Hz or non-finite
    "static",          # var.max() < min_variance (no motion at all)
    "rootSpeed",       # root translation speed > max_root_speed (m/s)
    "rootAccel",       # root translation accel  > max_accel    (m/s^2)
    "jointSnap",       # joint angular vel       > max_joint_rotvel (rad/s)
    "loadFail",        # could not load the file at all (IO, missing keys, NaN raw)
    "tooShortRaw",     # raw shape[0] < 4 frames before any processing
)

def diagnose_clip(motion: np.ndarray, fps: float, fkw: dict) -> list[str]:
    """Return the list of criteria the clip violates (empty -> passes).

    Mirrors quality_filter exactly but evaluates every check independently.
    """
    fails: list[str] = []
    T = motion.shape[0]

    if T < fkw["min_frames"]:
        fails.append("tooShort")

    if not np.isfinite(motion).all():
        fails.append("nanInf")
        return fails  # downstream checks meaningless on NaN

    if not np.isfinite(fps) or fps <= 1.0:
        fails.append("fpsBad")

    if motion.var(axis=0).max() < fkw["min_variance"]:
        fails.append("static")

    if motion.shape[1] >= 6 and T >= 2:
        trans = motion[:, 3:6]
        dt = 1.0 / 30.0  # post-resample fps; matches production filter
        velocity = np.diff(trans, axis=0) / dt
        speed = np.linalg.norm(velocity, axis=1)

        if speed.size and speed.max() > fkw["max_root_speed"]:
            fails.append("rootSpeed")

        if T >= 3:
            accel = np.diff(velocity, axis=0) / dt

            if np.linalg.norm(accel, axis=1).max() > fkw["max_accel"]:
                fails.append("rootAccel")

    if motion.shape[1] >= 9 and T >= 2:
        # Geodesic angular velocity on the rotation manifold (matches quality_filter).
        if geodesic_joint_vel(motion, fps) > fkw["max_joint_rotvel"]:
            fails.append("jointSnap")

    return fails

def run_amass(data_dir: str, fkw: dict, max_samples: int | None) -> dict:
    loader = AMASSLoader(data_dir)
    files = loader.discover_files()

    if max_samples is not None:
        files = files[:max_samples]
    log.info("[amass] %d candidate files", len(files))

    counts: dict = defaultdict(int)
    counts["total"] = len(files)
    per_reason: dict[str, int] = {r: 0 for r in REASONS}
    multi_reason: int = 0

    t0 = time.time()
    for i, fpath in enumerate(files):
        if i and i % 1000 == 0:
            log.info("[amass] %d/%d  passed=%d  rejected=%d (%.1fs)",
                     i, len(files), counts["passed"], counts["rejected"],
                     time.time() - t0)
        s = loader.load_file(fpath)

        if s is None:
            counts["rejected"] += 1
            per_reason["loadFail"] += 1
            continue

        if s.motion.shape[0] < 4:
            counts["rejected"] += 1
            per_reason["tooShortRaw"] += 1
            continue

        motion, fps = s.motion, float(s.fps)
        trim_s, trim_e = detect_tpose(motion)

        if trim_s > 0 or trim_e > 0:
            end = motion.shape[0] - trim_e if trim_e > 0 else motion.shape[0]
            motion = motion[trim_s:end]
            counts["tposeTrimmed"] += 1

        if abs(fps - 30.0) >= 0.5:
            motion = resample_to_fps(motion, fps, 30.0)
            counts["resampled"] += 1

        fails = diagnose_clip(motion, 30.0, fkw)

        if fails:
            counts["rejected"] += 1
            for r in fails:
                per_reason[r] += 1
            if len(fails) > 1:
                multi_reason += 1
        else:
            counts["passed"] += 1

    counts["multiReasonRejects"] = multi_reason
    counts["per_reason"] = dict(per_reason)
    counts["elapsedSec"] = round(time.time() - t0, 1)
    return counts

def run_humanml3d(hml_dir: str, amass_dir: str, fkw: dict, max_samples: int | None) -> dict:
    loader = HumanML3DLoader(hml_dir)
    amass = AMASSLoader(amass_dir)
    norm_map = build_norm_map(amass.discover_files(), amass.data_dir)

    all_texts = {item["clip_id"]: item["texts"] for item in loader.load_texts()}
    all_split_ids: set[str] = set()
    for split in ("train", "val", "test"):
        try:
            all_split_ids.update(loader.load_split(split))
        except FileNotFoundError:
            log.debug("[hml3d] split %s missing", split)
    idx_f = Path(hml_dir) / "index.csv"
    samples = parse_index_csv(idx_f, all_split_ids, all_texts, norm_map)

    if max_samples is not None:
        samples = samples[:max_samples]
    log.info("[hml3d] %d candidate samples (after split filter)", len(samples))

    cache, bad = preload_smplx(samples, Path(amass_dir))
    counts: dict = defaultdict(int)
    counts["total"] = len(samples)
    counts["preloadBad"] = len(bad or [])
    per_reason: dict[str, int] = {r: 0 for r in REASONS}
    multi_reason: int = 0

    bad_set = set(bad or [])
    samples = [s for s in samples if s["clip_id"] not in bad_set]

    t0 = time.time()
    for i, s in enumerate(samples):
        if i and i % 2000 == 0:
            log.info("[hml3d] %d/%d  passed=%d  rejected=%d (%.1fs)",
                     i, len(samples), counts["passed"], counts["rejected"],
                     time.time() - t0)
        clip_id = s["clip_id"]
        motion = cache.get(clip_id)

        if motion is None:
            counts["rejected"] += 1
            per_reason["loadFail"] += 1
            continue

        if motion.shape[0] < 4:
            counts["rejected"] += 1
            per_reason["tooShortRaw"] += 1
            continue

        # HumanML3D pose_data is already at 30 fps; no resample step in production.
        # detect_tpose is also not run on hml3d in production (clips are pre-trimmed).
        fails = diagnose_clip(motion, 30.0, fkw)

        if fails:
            counts["rejected"] += 1
            for r in fails:
                per_reason[r] += 1
            if len(fails) > 1:
                multi_reason += 1
        else:
            counts["passed"] += 1

    counts["multiReasonRejects"] = multi_reason
    counts["per_reason"] = dict(per_reason)
    counts["elapsedSec"] = round(time.time() - t0, 1)
    return counts

def print_report(label: str, stats: dict, fkw: dict) -> None:
    total = stats["total"]
    passed = stats.get("passed", 0)
    rejected = stats.get("rejected", 0)
    print()
    print(f"{'=' * 72}")
    print(f"  Quality stats: {label}")
    print(f"{'=' * 72}")
    print(
        f"  Thresholds: min_frames={fkw['min_frames']}  "
        f"max_root_speed={fkw['max_root_speed']} m/s  "
        f"max_accel={fkw['max_accel']} m/s^2"
    )
    print(f"              max_joint_rotvel={fkw['max_joint_rotvel']} rad/s  "
          f"min_variance={fkw['min_variance']}")
    print(f"  Total candidates    : {total}")
    print(f"  Passed              : {passed}  ({passed / max(total, 1) * 100:.1f} %)")
    print(f"  Rejected            : {rejected}  ({rejected / max(total, 1) * 100:.1f} %)")
    if "tposeTrimmed" in stats:
        print(f"  T-pose trimmed      : {stats['tposeTrimmed']}")
    if "resampled" in stats:
        print(f"  FPS resampled       : {stats['resampled']}")
    if stats.get("preloadBad"):
        print(f"  Preload failures    : {stats['preloadBad']}")
    print(f"  Elapsed             : {stats.get('elapsedSec', 0)} s")
    print()
    print("  Reason             Count  % of rejected  % of total")
    print(f"  {'-' * 56}")
    for reason in REASONS:
        n = stats["per_reason"].get(reason, 0)
        if n == 0:
            continue
        pct_rej = n / max(rejected, 1) * 100
        pct_tot = n / max(total, 1) * 100
        print(f"  {reason:<18} {n:>5}        {pct_rej:>5.1f} %        {pct_tot:>5.1f} %")
    multi = stats.get("multiReasonRejects", 0)
    if multi:
        print(f"  (clips failing >1 criterion: {multi})")
    print(f"{'=' * 72}")

def main() -> int:
    p = argparse.ArgumentParser(description="Per-criterion data-quality rejection stats")
    p.add_argument("--source", choices=["amass", "humanml3d", "all"], default="amass")
    p.add_argument("--amass-dir", default="data/amass", dest="amass_dir")
    p.add_argument("--humanml3d-dir", default=str(HML3D_DIR), dest="humanml3d_dir")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                   help="Cap the number of clips inspected (smoke test)")
    p.add_argument("--min-frames", type=int, default=DEFAULT_FILTER_KWARGS["min_frames"],
                   dest="min_frames")
    p.add_argument("--max-root-speed", type=float,
                   default=DEFAULT_FILTER_KWARGS["max_root_speed"], dest="max_root_speed")
    p.add_argument("--max-accel", type=float,
                   default=DEFAULT_FILTER_KWARGS["max_accel"], dest="max_accel")
    p.add_argument("--max-joint-rotvel", type=float,
                   default=DEFAULT_FILTER_KWARGS["max_joint_rotvel"], dest="max_joint_rotvel")
    p.add_argument("--min-variance", type=float,
                   default=DEFAULT_FILTER_KWARGS["min_variance"], dest="min_variance")
    p.add_argument("--output", default=None,
                   help="Optional JSON path to dump the full report")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    fkw = {
        "min_frames": args.min_frames,
        "max_root_speed": args.max_root_speed,
        "max_accel": args.max_accel,
        "max_joint_rotvel": args.max_joint_rotvel,
        "min_variance": args.min_variance,
    }

    report: dict = {"thresholds": fkw}

    if args.source in ("amass", "all"):
        report["amass"] = run_amass(args.amass_dir, fkw, args.max_samples)
        print_report("AMASS", report["amass"], fkw)

    if args.source in ("humanml3d", "all"):
        report["humanml3d"] = run_humanml3d(
            args.humanml3d_dir, args.amass_dir, fkw, args.max_samples,
        )
        print_report("HumanML3D", report["humanml3d"], fkw)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Report written to %s", args.output)

    return 0

if __name__ == "__main__":
    sys.exit(main())
