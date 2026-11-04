# python scripts/research/segment_coverage.py --pack data/amass_tokens_fsq8x1024.npz --lengths 1 2 4 --top 1000
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def segment_counts(windows: list[np.ndarray], length: int) -> Counter:
    counts: Counter = Counter()
    for codes in windows:
        for start in range(codes.shape[0] - length + 1):
            counts[codes[start : start + length].tobytes()] += 1
    return counts


def top_share(counts: Counter, top: int) -> float:
    total = sum(counts.values())
    kept = sum(count for _, count in counts.most_common(top))
    return kept / total


def check_self() -> None:
    windows = [np.array([[1, 1], [1, 1], [2, 2]], dtype=np.int16)]
    counts = segment_counts(windows, 2)
    assert len(counts) == 2
    assert top_share(counts, 1) == 0.5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("data/amass_tokens_fsq8x1024.npz"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--top", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    check_self()
    pack = np.load(arguments.pack, allow_pickle=False)
    windows = [pack[key] for key in pack.files]
    print(f"{len(windows)} windows from {arguments.pack}")
    for length in arguments.lengths:
        counts = segment_counts(windows, length)
        total = sum(counts.values())
        distinct = len(counts)
        share = top_share(counts, arguments.top)
        print(
            f"length {length}: {total} segments, {distinct} distinct ({distinct / total:.1%}), "
            f"top {arguments.top} cover {share:.2%}"
        )


if __name__ == "__main__":
    main()
