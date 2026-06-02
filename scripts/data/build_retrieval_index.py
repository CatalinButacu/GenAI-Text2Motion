#!/usr/bin/env python
"""Build a retrieval index from a list of motion prompts (US-09).

Reads prompts from a plain-text file (one per line), encodes them with SBERT,
and saves the index as a .npz file consumable by MotionRetriever.

Usage
-----
  python scripts/data/build_retrieval_index.py \\
      --prompts data/humanml3d/prompts.txt \\
      --output data/retrieval_index.npz

  # Or from a JSON file containing a list of prompt strings:
  python scripts/data/build_retrieval_index.py \\
      --prompts data/eval/prompt_following_suite.json --json-key prompt \\
      --output data/retrieval_index.npz
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.modules.motion.augment import MotionRetriever

log = logging.getLogger(__name__)


def load_prompts(promptsPath: str, jsonKey: str | None = None) -> list[str]:
    p = Path(promptsPath)

    if p.suffix == ".json":
        if jsonKey is None:
            raise ValueError("--json-key required when loading from a JSON file")

        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [str(entry[jsonKey]) for entry in data if jsonKey in entry]

        raise ValueError(f"JSON file must contain a list of objects, got {type(data)}")

    lines = p.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Build SBERT retrieval index for motion prompts")
    p.add_argument("--prompts", required=True, help="Path to .txt or .json prompt file")
    p.add_argument(
        "--json-key",
        dest="json_key",
        default=None,
        help="Key to extract from JSON objects (required for JSON input)",
    )
    p.add_argument(
        "--output",
        default="data/retrieval_index.npz",
        help="Output path for the .npz index (default: data/retrieval_index.npz)",
    )
    p.add_argument(
        "--sbert-model",
        dest="sbert_model",
        default="all-MiniLM-L6-v2",
        help="SBERT model name (default: all-MiniLM-L6-v2)",
    )
    args = p.parse_args()

    prompts = load_prompts(args.prompts, jsonKey=args.json_key)
    log.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    MotionRetriever.build_index(prompts, output_path=args.output, sbert_model=args.sbert_model)
    log.info("Done. Index saved to %s", args.output)


if __name__ == "__main__":
    main()
