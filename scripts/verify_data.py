#!/usr/bin/env python
"""Verify data integrity. Hash AMASS + HumanML3D; refuse training if corpus drifted.

Usage:
    python scripts/verify_data.py           # record baseline (first time)
    python scripts/verify_data.py --check   # compare against baseline, exit 1 on mismatch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.shared.run_ctx import dataFingerprint

DATA_ROOTS = ["data/AMASS", "data/humanml3d", "data/arctic/unpack"]
BASELINE = Path("data/.cache/data_fingerprint.json")

def compute() -> dict[str, str]:
    result = {}

    for r in DATA_ROOTS:
        if Path(r).exists():
            result[r] = dataFingerprint([r])
            print(f"  {r:30s}  {result[r]}")
        else:
            print(f"  {r:30s}  (missing)")

    return result

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="Compare vs baseline, exit 1 on mismatch")
    args = p.parse_args()

    print("[verify_data] computing fingerprints ...")
    current = compute()

    if args.check:
        if not BASELINE.exists():
            print(f"[verify_data] no baseline at {BASELINE} -- run without --check first")
            sys.exit(2)

        prev = json.loads(BASELINE.read_text())
        diffs = [k for k in set(prev) | set(current) if prev.get(k) != current.get(k)]

        if diffs:
            print(f"[verify_data] FAIL: fingerprint drift in {diffs}")
            print(f"  expected: {json.dumps(prev, indent=2)}")
            print(f"  current:  {json.dumps(current, indent=2)}")
            sys.exit(1)
        print("[verify_data] OK: data matches baseline")
    else:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2))
        print(f"[verify_data] baseline written -> {BASELINE}")

if __name__ == "__main__":
    main()
