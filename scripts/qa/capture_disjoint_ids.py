# python scripts/qa/capture_disjoint_ids.py
from __future__ import annotations

import csv
import random
from pathlib import Path

from text2motion.motion.contracts import Split
from text2motion.motion.storage import SPLIT_FILES, listed_clip_ids

ROOT = Path("data/HumanML3D_official")
OUT = Path("doc/notes")
SEED = 2026
POPULATION = (("val", Split.VALIDATION, False), ("test", Split.TEST, True))


def capture_of(index_path: Path) -> dict[str, str]:
    with index_path.open(encoding="utf-8") as handle:
        return {
            row["new_name"].removesuffix(".npy"): str(Path(row["source_path"].strip())).casefold()
            for row in csv.DictReader(handle)
        }


def base_id(clip_id: str) -> str:
    return clip_id.removeprefix("M")


def main() -> None:
    captures = capture_of(ROOT / "index.csv")
    train = {base_id(clip) for clip in listed_clip_ids(ROOT / SPLIT_FILES[Split.TRAIN])}
    train_captures = {captures[clip] for clip in train}
    rows = []
    for name, split, with_mirrors in POPULATION:
        listed = listed_clip_ids(ROOT / SPLIT_FILES[split])
        population = listed if with_mirrors else [clip for clip in listed if clip == base_id(clip)]
        disjoint = [clip for clip in population if captures[base_id(clip)] not in train_captures]
        matched = sorted(random.Random(SEED).sample(population, len(disjoint)))
        (OUT / f"capture_disjoint_{name}.txt").write_text("\n".join(disjoint) + "\n", "utf-8")
        (OUT / f"capture_matched_random_{name}.txt").write_text("\n".join(matched) + "\n", "utf-8")
        rows.append([name, len(population), len(disjoint), len(population) - len(disjoint)])
    with (OUT / "capture_disjoint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "clips", "disjoint_clips", "shared_capture_clips"])
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
