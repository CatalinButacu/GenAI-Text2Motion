# python scripts/qa/rare_verb_kind_probe.py --out docs/paper/pics/rare_verb_kind.csv
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from verb_generation_probe import CONFIG, SEED, chosen_verbs, real_clips

from text2motion.app.bootstrap import ApplicationBootstrap

RARE_CLIPS = 150
RARE_SAMPLES = 32
FREQUENT_SAMPLES = 64
MIN_RARE_WORD_VERBS = 10
RARE_WORD = "rare_word"
RARE_MOTION = "rare_motion"


def centroid_radius(embeddings: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = embeddings.mean(axis=0)
    return centroid, float(np.median(np.linalg.norm(embeddings - centroid, axis=1)))


def nearest_cover(
    embeddings: np.ndarray, frequent: dict[str, tuple[np.ndarray, float]]
) -> tuple[str, float]:
    ratios = {
        verb: float(np.median(np.linalg.norm(embeddings - centroid, axis=1))) / radius
        for verb, (centroid, radius) in frequent.items()
    }
    verb = min(ratios, key=ratios.__getitem__)
    return verb, ratios[verb]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbs", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("docs/paper/pics/rare_verb_kind.csv"))
    arguments = parser.parse_args()

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    downsample = app.config.tokenizer.downsample
    evaluator = app.create_humanml3d_generation_evaluator()
    verbs = chosen_verbs(arguments.verbs)
    rare = [(verb, clips) for verb, clips in verbs if clips < RARE_CLIPS]
    common = [verb for verb, clips in verbs if clips >= RARE_CLIPS]
    common_clips = real_clips(common, downsample, FREQUENT_SAMPLES)
    frequent = {
        verb: centroid_radius(evaluator.compute_motion_embeddings(common_clips[verb]))
        for verb in common
    }
    rare_clips = real_clips([verb for verb, _ in rare], downsample, RARE_SAMPLES)

    rows = []
    for verb, clips in rare:
        embeddings = evaluator.compute_motion_embeddings(rare_clips[verb])
        cover, ratio = nearest_cover(embeddings, frequent)
        rows.append(
            {
                "verb": verb,
                "train_clips": clips,
                "clips_used": len(rare_clips[verb]),
                "nearest_frequent_verb": cover,
                "distance_ratio": ratio,
                "kind": RARE_WORD if ratio <= 1.0 else RARE_MOTION,
            }
        )
    with arguments.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rare_words = sum(row["kind"] == RARE_WORD for row in rows)
    for row in sorted(rows, key=lambda row: row["distance_ratio"]):
        print(row)
    print(f"rare_word {rare_words} of {len(rows)}; frequent verbs {len(common)}")
    print("continue to step 2" if rare_words >= MIN_RARE_WORD_VERBS else "stop after step 1")


if __name__ == "__main__":
    main()
