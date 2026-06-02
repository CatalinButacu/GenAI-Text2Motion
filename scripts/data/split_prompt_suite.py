#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation.prompt_suite_schema import split_golden_canary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split prompt suite into golden and canary sets")
    parser.add_argument("--suite", default="data/eval/prompt_following_suite.json")
    parser.add_argument("--golden", default="data/eval/prompt_following_golden.json")
    parser.add_argument("--canary", default="data/eval/prompt_following_canary.json")
    parser.add_argument("--canary-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    suite_path = Path(args.suite)
    rows = json.loads(suite_path.read_text(encoding="utf-8"))
    golden, canary = split_golden_canary(
        rows,
        canary_size=args.canary_size,
        seed=args.seed,
    )

    golden_path = Path(args.golden)
    canary_path = Path(args.canary)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    canary_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    canary_path.write_text(json.dumps(canary, indent=2), encoding="utf-8")

    print(f"golden={len(golden)} canary={len(canary)}")


if __name__ == "__main__":
    main()
