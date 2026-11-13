# python scripts/qa/action_group_motion_check.py --out doc/notes/action_group_merges.csv
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from action_vocabulary import ACTION, INVENTORY, LABELS, TYPO, merged_labels, read_labels
from verb_generation_probe import CONFIG, SEED, real_clips

from text2motion.app.bootstrap import ApplicationBootstrap

THRESHOLD = 1.10
SAMPLES = 32
MIN_POSITIVE = 0.8
MAX_NEGATIVE = 0.1
GROUPED = (ACTION, TYPO)


def merge_ratio(own: np.ndarray, group: np.ndarray) -> float:
    count = len(own)
    leave_one_out = (own.sum(axis=0, keepdims=True) - own) / (count - 1)
    own_distance = float(np.median(np.linalg.norm(own - leave_one_out, axis=1)))
    group_distance = float(np.median(np.linalg.norm(own - group.mean(axis=0), axis=1)))
    return group_distance / own_distance


def worker_groups(
    path: Path, statuses: dict[str, str]
) -> tuple[dict[str, str], dict[str, set[str]]]:
    labels = read_labels(path)
    groups: dict[str, set[str]] = defaultdict(set)
    for verb, row in labels.items():
        if row["group"] and statuses.get(verb) in GROUPED:
            groups[row["group"]].add(verb)
    return {verb: row["group"] for verb, row in labels.items()}, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("doc/notes/action_group_merges.csv"))
    arguments = parser.parse_args()

    rows = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    verbs = [row["verb"] for row in rows if int(row["train_clips"]) >= 5]
    merged = merged_labels(verbs)
    statuses = {verb: status for verb, (status, _, _) in merged.items()}

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    evaluator = app.create_humanml3d_generation_evaluator()
    grouped = [verb for verb in verbs if statuses[verb] in GROUPED]
    clips = real_clips(grouped, app.config.tokenizer.downsample, SAMPLES)
    embeddings = {
        verb: evaluator.compute_motion_embeddings(clips[verb])
        for verb in grouped
        if len(clips[verb]) >= 2
    }

    def stacked(members: set[str]) -> np.ndarray | None:
        present = [embeddings[verb] for verb in sorted(members) if verb in embeddings]
        return np.concatenate(present) if present else None

    results = []
    agreed: dict[str, set[str]] = defaultdict(set)
    for verb, (status, key, source) in merged.items():
        if source == "agree" and key and status in GROUPED:
            agreed[key].add(verb)
    rng = np.random.default_rng(SEED)
    multi = sorted(key for key, members in agreed.items() if len(members) > 1)
    for key in multi:
        for verb in sorted(agreed[key]):
            rest = stacked(agreed[key] - {verb})
            if verb not in embeddings or rest is None:
                continue
            ratio = merge_ratio(embeddings[verb], rest)
            results.append(
                [verb, "positive", "agreed", " ".join(sorted(agreed[key] - {verb})), ratio]
            )
            other = multi[rng.integers(len(multi))]
            while other == key:
                other = multi[rng.integers(len(multi))]
            wrong = stacked(agreed[other])
            if wrong is not None:
                ratio = merge_ratio(embeddings[verb], wrong)
                results.append([verb, "negative", "random", " ".join(sorted(agreed[other])), ratio])

    workers = [worker_groups(path, statuses) for path in LABELS]
    for verb, (_, _, source) in merged.items():
        if source != "split" or verb not in embeddings:
            continue
        for path, (labels, groups) in zip(LABELS, workers, strict=True):
            members = groups.get(labels.get(verb, ""), set()) - {verb}
            group = stacked(members)
            if group is None:
                continue
            ratio = merge_ratio(embeddings[verb], group)
            results.append([verb, "split", path.stem, " ".join(sorted(members)), ratio])

    with arguments.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["verb", "kind", "source", "group_members", "ratio", "joins"])
        for verb, kind, source, members, ratio in results:
            writer.writerow([verb, kind, source, members, f"{ratio:.4f}", ratio <= THRESHOLD])

    def accepted(kind: str) -> float:
        ratios = [row[4] for row in results if row[1] == kind]
        return float(np.mean([ratio <= THRESHOLD for ratio in ratios])) if ratios else float("nan")

    positive, negative = accepted("positive"), accepted("negative")
    valid = positive >= MIN_POSITIVE and negative <= MAX_NEGATIVE
    print(f"positive control accepted {positive:.1%}; negative control accepted {negative:.1%}")
    print(
        f"split proposals {sum(row[1] == 'split' for row in results)}, accepted {accepted('split'):.1%}"
    )
    print("test VALID" if valid else "test NOT VALID: keep the disputed verbs apart")


if __name__ == "__main__":
    main()
