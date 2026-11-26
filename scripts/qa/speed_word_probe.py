# python scripts/qa/speed_word_probe.py --out runs/outputs/qa/speed_word_val.json --csv docs/paper/pics/speed_word_probe.csv
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from text2motion.app.runtime.context import ApplicationBootstrap
from text2motion.evaluation.metrics import paired_mean_ratio
from text2motion.generation.contracts import Backbone, SamplingConfig
from text2motion.motion.recovery import recover_skeleton
from text2motion.motion.representation import FPS

SEED = 2026
CONFIG = "configs/generator/size_sweep/width_320.yaml"
TOKENIZER = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
GENERATORS = (
    (Backbone.TRANSFORMER, "runs/checkpoints/generator/generator_transformer_w320_p1.pt"),
    (Backbone.MAMBA, "runs/checkpoints/generator/generator_mamba_w320_p1.pt"),
)
VERBS = (
    ("walks", "many"),
    ("jumps", "few"),
    ("turns", "few"),
    ("bows", "rare"),
    ("hops", "rare"),
    ("spins", "rare"),
    ("climbs", "rare"),
)
WORDS = ("slowly", "quickly")
SAMPLES = 16
TOKEN_STEPS = 30
RESAMPLES = 1000
SAMPLING = SamplingConfig(temperature=1.0, top_p=0.9, cfg_scale=6.0)


def speeds(joints: np.ndarray) -> tuple[float, float]:
    local = joints - joints[:, :1]
    limb = np.linalg.norm(np.diff(local, axis=0), axis=-1).mean() * FPS
    root = np.linalg.norm(np.diff(joints[:, 0, [0, 2]], axis=0), axis=-1).mean() * FPS
    return float(limb), float(root)


def prompt_speeds(pipeline, resources, caption: str) -> tuple[np.ndarray, np.ndarray]:
    seeds = [SEED + index for index in range(SAMPLES)]
    with torch.no_grad():
        batch = pipeline.generate_batch([caption] * SAMPLES, TOKEN_STEPS, SAMPLING, seeds=seeds)
    if batch is None:
        raise RuntimeError(f"the generator returned no clip for {caption!r}")
    values = [
        speeds(recover_skeleton(resources.denormalize(item.motion.features.cpu().numpy())))
        for item in batch
    ]
    return np.array([v[0] for v in values]), np.array([v[1] for v in values])


def verb_rows(pipeline, resources, backbone: Backbone, verb: str, group: str) -> list[dict]:
    plain_limb, plain_root = prompt_speeds(pipeline, resources, f"a person {verb}")
    rows = []
    for word in WORDS:
        limb, root = prompt_speeds(pipeline, resources, f"a person {verb} {word}")
        for name, value, base in (("limb", limb, plain_limb), ("root", root, plain_root)):
            ratio, low, high = paired_mean_ratio(value, base, RESAMPLES, SEED)
            rows.append(
                {
                    "backbone": backbone.value,
                    "verb": verb,
                    "group": group,
                    "word": word,
                    "speed": name,
                    "plain_m_per_s": float(base.mean()),
                    "word_m_per_s": float(value.mean()),
                    "ratio": ratio,
                    "ci_lo": low,
                    "ci_hi": high,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/outputs/qa/speed_word_val.json"))
    parser.add_argument("--csv", type=Path, default=Path("docs/paper/pics/speed_word_probe.csv"))
    arguments = parser.parse_args()

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    tokenizer = app.models.load_tokenizer(TOKENIZER)
    resources = app.create_humanml3d_generation_evaluator().resources
    rows: list[dict] = []
    for backbone, checkpoint in GENERATORS:
        pipeline = app.models.load_text_to_motion_generator(
            backbone, checkpoint, tokenizer, tokenizer_checkpoint=TOKENIZER
        ).eval()
        for verb, group in VERBS:
            for row in verb_rows(pipeline, resources, backbone, verb, group):
                print(json.dumps(row), flush=True)
                rows.append(row)
        del pipeline
        torch.cuda.empty_cache()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    arguments.csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
