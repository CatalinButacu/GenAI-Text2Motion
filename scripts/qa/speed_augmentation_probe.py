# python scripts/qa/speed_augmentation_probe.py --out doc/paper/pics/speed_augmentation_validation.csv
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from a15_v3_verdict import paired_interval
from scipy import interpolate
from verb_generation_probe import CONFIG, MAX_FRAMES, SEED, chosen_verbs, real_clips, retrieval

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.evaluation.embeddings import compute_guo_text_embeddings
from text2motion.motion.representation import SLICES

SPEEDS = (0.8, 0.9, 1.1, 1.2)
RARE_CLIPS = 150
SAMPLES = 32
MIN_DROP = -0.05
ROOT_VELOCITY = slice(0, 3)


def speed_changed(motion: np.ndarray, speed: float, downsample: int) -> np.ndarray:
    frames = len(motion)
    new_frames = max(downsample, round(frames / speed))
    source_times = np.linspace(0.0, frames - 1, new_frames)
    spline = interpolate.make_interp_spline(np.arange(frames), motion, k=1, axis=0)
    changed = spline(source_times).astype(np.float32)
    changed[:, ROOT_VELOCITY] *= speed
    changed[:, SLICES["vel"]] *= speed
    changed[:, SLICES["foot"]] = (changed[:, SLICES["foot"]] >= 0.5).astype(np.float32)
    kept = (min(len(changed), MAX_FRAMES) // downsample) * downsample
    return changed[:kept]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbs", type=int, default=100)
    parser.add_argument(
        "--out", type=Path, default=Path("doc/paper/pics/speed_augmentation_validation.csv")
    )
    arguments = parser.parse_args()

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    downsample = app.config.tokenizer.downsample
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    verbs = chosen_verbs(arguments.verbs)
    names = [verb for verb, _ in verbs]
    rare = [verb for verb, clips in verbs if clips < RARE_CLIPS]
    text_emb = compute_guo_text_embeddings(
        resources.text_embedder,
        [
            resources.build_guo_text_inputs(["a/DET", "person/NOUN", f"{verb}/VERB"])
            for verb in names
        ],
        resources.device,
    )
    real = real_clips(rare, downsample, SAMPLES)
    top5 = {speed: {} for speed in (1.0, *SPEEDS)}
    for verb in rare:
        own = names.index(verb)
        for speed in top5:
            motions = [
                clip if speed == 1.0 else speed_changed(clip, speed, downsample)
                for clip in real[verb]
            ]
            _, rate = retrieval(evaluator.compute_motion_embeddings(motions), text_emb, own)
            top5[speed][verb] = rate

    rows = []
    for speed in SPEEDS:
        differences = np.array([top5[speed][verb] - top5[1.0][verb] for verb in rare])
        low, high = paired_interval(differences)
        rows.append(
            {
                "speed": speed,
                "rare_verbs": len(rare),
                "original_top5": float(np.mean([top5[1.0][verb] for verb in rare])),
                "changed_top5": float(np.mean([top5[speed][verb] for verb in rare])),
                "difference": float(differences.mean()),
                "difference_low": low,
                "difference_high": high,
                "keeps_action": low > MIN_DROP,
            }
        )
    with arguments.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
