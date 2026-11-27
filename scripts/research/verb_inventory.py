# python scripts/research/verb_inventory.py --out docs/notes/verb_inventory.csv
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path("data/processed/humanml3d")
SPLITS = ("train", "val", "test")
NOT_ACTIONS = frozenset(
    {"be", "have", "do", "seem", "appear", "begin", "start", "continue", "try", "use"}
)


def caption_verbs(line: str) -> set[str]:
    parts = line.split("#")
    if len(parts) < 2:
        raise ValueError(f"caption line without a tag field: {line!r}")
    verbs = set()
    for token in parts[1].split():
        word, _, tag = token.rpartition("/")
        if tag == "VERB" and word and word not in NOT_ACTIONS:
            verbs.add(word.lower())
    return verbs


def split_counts(split: str) -> tuple[Counter, Counter]:
    clips: Counter = Counter()
    captions: Counter = Counter()
    for clip_id in (ROOT / f"{split}.txt").read_text(encoding="utf-8").split():
        if clip_id.startswith("M"):
            continue
        path = ROOT / "texts" / f"{clip_id}.txt"
        if not path.exists():
            continue
        verbs_per_clip: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            verbs = caption_verbs(line)
            captions.update(verbs)
            verbs_per_clip |= verbs
        clips.update(verbs_per_clip)
    return clips, captions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/notes/verb_inventory.csv"))
    parser.add_argument("--min_train_clips", type=int, default=5)
    arguments = parser.parse_args()
    counts = {split: split_counts(split) for split in SPLITS}
    train_clips = counts["train"][0]
    verbs = sorted(
        (verb for verb, n in train_clips.items() if n >= arguments.min_train_clips),
        key=lambda verb: -train_clips[verb],
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["verb", *(f"{s}_{k}" for s in SPLITS for k in ("clips", "captions"))])
        for verb in verbs:
            writer.writerow([verb, *(counts[s][i][verb] for s in SPLITS for i in (0, 1))])
    print(f"{len(verbs)} verbs with >= {arguments.min_train_clips} train clips -> {arguments.out}")


if __name__ == "__main__":
    main()
