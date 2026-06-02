#!/usr/bin/env python
"""Data quality gate — fail fast before training or evaluation.

Checks a sample of clips from a data directory and validates normalization
stats. Returns exit code 0 on all-pass, 1 on any violation.

Usage
-----
  python scripts/data/quality_gate.py --data-dir data/amass --stats-dir data/stats
  python scripts/data/quality_gate.py --data-dir data/amass --stats-dir data/stats \
      --output results/quality_gate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

POSE_DIM = 168
MIN_FRAMES = 16
PELVIS_Z_IDX = 5
COORD_SYSTEM_Z_MEAN_MIN = 0.0  # Y-up: pelvis is above zero
GLOB_NPY = "**/*.npy"
GLOB_NPZ = "**/*.npz"

log = logging.getLogger(__name__)


def check_files_exist(dataDir: str) -> dict:
    p = Path(dataDir)
    files = list(p.glob(GLOB_NPY)) + list(p.glob(GLOB_NPZ))
    passed = len(files) > 0
    return {
        "name": "files_exist",
        "passed": passed,
        "detail": f"{len(files)} motion files found in {dataDir}",
    }


def loadFirstArray(f) -> np.ndarray | None:
    try:
        if f.suffix == ".npz":
            data = np.load(str(f))
            arr = data[next(iter(data))]
        else:
            arr = np.load(str(f))

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return arr

    except Exception as exc:
        log.warning("Could not load %s: %s", f.name, exc)
        return None


def check_sample_quality(dataDir: str, sampleSize: int, seed: int) -> dict:
    p = Path(dataDir)
    all_files = sorted(p.glob(GLOB_NPY)) + sorted(p.glob(GLOB_NPZ))

    if not all_files:
        return {"name": "sample_quality", "passed": False, "detail": "no files to sample"}

    rng = random.Random(seed)
    sampled = rng.sample(all_files, min(sampleSize, len(all_files)))

    nan_count = 0
    shape_bad = 0
    short_count = 0
    checked = 0

    for f in sampled:
        arr = loadFirstArray(f)

        if arr is None:
            shape_bad += 1
            continue

        checked += 1

        if not np.isfinite(arr).all():
            nan_count += 1

        if arr.shape[-1] != POSE_DIM:
            shape_bad += 1

        if arr.shape[0] < MIN_FRAMES:
            short_count += 1

    issues = []

    if nan_count:
        issues.append(f"{nan_count} files have NaN/Inf")

    if shape_bad:
        issues.append(f"{shape_bad} files have wrong pose dim (expected {POSE_DIM})")

    if short_count:
        issues.append(f"{short_count} files have < {MIN_FRAMES} frames")

    passed = len(issues) == 0
    detail = f"Checked {checked}/{len(all_files)} files. " + (
        "; ".join(issues) if issues else "All OK."
    )
    return {"name": "sample_quality", "passed": passed, "detail": detail}


def check_stats(statsDir: str) -> dict:
    p = Path(statsDir)
    issues = []

    for fname in ("mean.npy", "std.npy"):
        fpath = p / fname
        if not fpath.exists():
            issues.append(f"missing {fname}")
            continue
        arr = np.load(str(fpath))
        if arr.shape != (POSE_DIM,):
            issues.append(f"{fname} shape {arr.shape} != ({POSE_DIM},)")

    passed = len(issues) == 0
    detail = "; ".join(issues) if issues else f"mean.npy and std.npy found with shape ({POSE_DIM},)"
    return {"name": "stats_files", "passed": passed, "detail": detail}


def check_coord_system(dataDir: str, sampleSize: int, seed: int) -> dict:
    p = Path(dataDir)
    all_files = sorted(p.glob(GLOB_NPY)) + sorted(p.glob(GLOB_NPZ))

    if not all_files:
        return {"name": "coord_system", "passed": True, "detail": "no files — skipped"}

    rng = random.Random(seed + 1)
    sampled = rng.sample(all_files, min(sampleSize // 4, len(all_files)))

    z_values = []

    for f in sampled:
        arr = loadFirstArray(f)

        if arr is not None and arr.shape[-1] >= PELVIS_Z_IDX + 1:
            z_values.append(float(arr[:, PELVIS_Z_IDX].mean()))

    if not z_values:
        return {"name": "coord_system", "passed": True, "detail": "could not extract Z — skipped"}

    mean_z = float(np.mean(z_values))
    passed = mean_z > COORD_SYSTEM_Z_MEAN_MIN
    detail = f"Mean pelvis-Z across sample = {mean_z:.4f} m (expected > 0 for Y-up)"
    return {"name": "coord_system", "passed": passed, "detail": detail}


def check_no_duplicates(dataDir: str) -> dict:
    p = Path(dataDir)
    all_files = list(p.glob("**/*.npy")) + list(p.glob("**/*.npz"))
    stems = [f.stem for f in all_files]
    dupe_count = len(stems) - len(set(stems))
    passed = dupe_count == 0
    detail = f"{dupe_count} duplicate file stems found" if dupe_count else "No duplicate stems"
    return {"name": "no_duplicates", "passed": passed, "detail": detail}


def run_gate(
    dataDir: str,
    statsDir: str,
    sampleSize: int = 200,
    seed: int = 42,
    outputPath: str | None = None,
) -> dict:
    """Run all quality checks.

    Returns
    -------
    dict with keys "passed" (bool) and "checks" (list of per-check dicts).
    """
    checks = [
        check_files_exist(dataDir),
        check_sample_quality(dataDir, sampleSize, seed),
        check_stats(statsDir),
        check_coord_system(dataDir, sampleSize, seed),
        check_no_duplicates(dataDir),
    ]

    passed = all(c["passed"] for c in checks)
    result = {"passed": passed, "checks": checks}

    print_gate_table(checks)

    if outputPath:
        os.makedirs(Path(outputPath).parent, exist_ok=True)
        Path(outputPath).write_text(
            __import__("json").dumps(result, indent=2), encoding="utf-8"
        )
        log.info("Gate report saved to %s", outputPath)

    return result


def print_gate_table(checks: list[dict]) -> None:
    print(f"\n{'Check':<25} {'Status':<8} Detail")
    print("-" * 80)

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  {c['name']:<23} {status:<8} {c['detail']}")

    print()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Data quality gate")
    p.add_argument("--data-dir", required=True, dest="data_dir")
    p.add_argument("--stats-dir", default="data/stats", dest="stats_dir")
    p.add_argument("--sample-size", type=int, default=200, dest="sample_size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    result = run_gate(
        dataDir=args.data_dir,
        statsDir=args.stats_dir,
        sampleSize=args.sample_size,
        seed=args.seed,
        outputPath=args.output,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
