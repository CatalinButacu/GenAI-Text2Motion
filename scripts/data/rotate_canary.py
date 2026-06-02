#!/usr/bin/env python
"""Canary rotation for the prompt-following evaluation suite.

Moves the lowest-diversity canary entries back to the golden pool and
draws fresh canary entries from an expansion file.

Usage
-----
  python scripts/data/rotate_canary.py \
      --suite data/eval/prompt_following_suite.json \
      --golden data/eval/prompt_following_golden.json \
      --canary data/eval/prompt_following_canary.json \
      --expansion data/eval/prompt_following_expansion.json \
      --canary-size 40 \
      --seed 42

The expansion file should be a JSON array of new prompt entries with the
same schema as prompt_following_suite.json.  If not provided or empty,
the script simply re-shuffles the existing canary using a new seed.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.evaluation.prompt_suite_schema import split_golden_canary, validate_suite

log = logging.getLogger(__name__)


def write_one_line(entries: list[dict], path: Path) -> None:
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    path.write_text("[\n" + ",\n".join(lines) + "\n]", encoding="utf-8")


def load_json_array(path: Path) -> list[dict]:
    if not path.exists():
        return []

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Rotate canary entries in the prompt-following suite")
    p.add_argument("--suite", default="data/eval/prompt_following_suite.json")
    p.add_argument("--golden", default="data/eval/prompt_following_golden.json")
    p.add_argument("--canary", default="data/eval/prompt_following_canary.json")
    p.add_argument("--expansion", default="", help="Optional file with new prompt entries to add")
    p.add_argument("--canary-size", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    suite_entries = load_json_array(Path(args.suite))
    log.info("Loaded %d suite entries", len(suite_entries))

    if args.expansion:
        expansion_path = Path(args.expansion)
        new_entries = load_json_array(expansion_path)
        log.info("Loaded %d expansion entries from %s", len(new_entries), expansion_path)
    else:
        new_entries = []

    combined = suite_entries + new_entries
    validated = validate_suite(combined, min_size=args.canary_size + 1)
    log.info("Combined pool: %d entries", len(validated))

    golden_entries, canary_entries = split_golden_canary(
        validated, canary_size=args.canary_size, seed=args.seed
    )

    golden_path = Path(args.golden)
    canary_path = Path(args.canary)

    write_one_line(golden_entries, golden_path)
    write_one_line(canary_entries, canary_path)

    log.info("Wrote golden=%d canary=%d", len(golden_entries), len(canary_entries))
    print(f"golden={len(golden_entries)} canary={len(canary_entries)}")


if __name__ == "__main__":
    main()
