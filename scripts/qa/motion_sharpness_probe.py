# python scripts/qa/motion_sharpness_probe.py --clips 256 --out runs/outputs/qa/motion_sharpness_val.json --csv docs/paper/pics/motion_sharpness_validation.csv
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from text2motion.app.runtime.context import ApplicationBootstrap
from text2motion.evaluation.metrics import paired_mean_ratio, sharpness_statistics
from text2motion.evaluation.samples import deterministic_generation_seed
from text2motion.generation.contracts import Backbone, SamplingConfig
from text2motion.motion.contracts import Split
from text2motion.motion.recovery import recover_skeleton

SEED = 2026
CONFIG = "configs/generator/size_sweep/width_320.yaml"
TOKENIZER = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
GENERATORS = (
    (Backbone.TRANSFORMER, "runs/checkpoints/generator/generator_transformer_w320_p1.pt"),
    (Backbone.MAMBA, "runs/checkpoints/generator/generator_mamba_w320_p1.pt"),
)
SAMPLING = SamplingConfig(temperature=1.0, top_p=0.9, cfg_scale=6.0)
METRICS = (
    "peak_speed_m_per_s",
    "peak_acceleration_m_per_s2",
    "mean_jerk_m_per_s3",
    "foot_skate_m_per_s",
)
MAX_FRAMES = 196
MIN_FRAMES = 40
BATCH = 16
RESAMPLES = 1000


def validation_clips(resources, downsample: int, count: int) -> list[tuple[str, np.ndarray, str]]:
    clip_ids = resources.clip_ids(Split.VALIDATION)
    random.Random(SEED).shuffle(clip_ids)
    vectors = resources.motion_dataset_dir / "new_joint_vecs"
    chosen = []
    for clip_id in clip_ids:
        if len(chosen) == count:
            break
        raw = np.load(vectors / f"{clip_id}.npy").astype(np.float32)
        frames = (min(len(raw), MAX_FRAMES) // downsample) * downsample
        if frames < MIN_FRAMES or not np.isfinite(raw).all():
            continue
        text = (resources.annotation_dir / f"{clip_id}.txt").read_text(encoding="utf-8")
        chosen.append((clip_id, raw[:frames], text.splitlines()[0].split("#")[0].strip()))
    if len(chosen) < count:
        raise ValueError(f"only {len(chosen)} usable validation clips, {count} requested")
    return chosen


def roundtrip_rows(tokenizer, resources, clips, device) -> list[dict[str, float]]:
    rows = []
    with torch.no_grad():
        for _, raw, _ in clips:
            normalized = torch.from_numpy(resources.normalize(raw)).unsqueeze(0).to(device)
            decoded = tokenizer.network.decode(tokenizer.network.encode(normalized))[0]
            rows.append(
                sharpness_statistics(recover_skeleton(resources.denormalize(decoded.cpu().numpy())))
            )
    return rows


def generated_rows(pipeline, resources, clips, downsample: int) -> list[dict[str, float]]:
    rows: list[dict[str, float] | None] = [None] * len(clips)
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, (_, raw, _) in enumerate(clips):
        by_length[len(raw) // downsample].append(index)
    with torch.no_grad():
        for steps, indices in by_length.items():
            for start in range(0, len(indices), BATCH):
                group = indices[start : start + BATCH]
                batch = pipeline.generate_batch(
                    [clips[i][2] for i in group],
                    steps,
                    SAMPLING,
                    seeds=[deterministic_generation_seed(clips[i][0]) for i in group],
                )
                if batch is None:
                    raise RuntimeError(f"the generator returned no clip for {steps} token steps")
                for index, item in zip(group, batch, strict=True):
                    features = resources.denormalize(item.motion.features.cpu().numpy())
                    rows[index] = sharpness_statistics(recover_skeleton(features))
    missing = [clips[i][0] for i, row in enumerate(rows) if row is None]
    if missing:
        raise RuntimeError(f"no generated motion for {missing[:10]}")
    return [row for row in rows if row is not None]


def summary(stages: dict[str, list[dict[str, float]]]) -> list[dict[str, float | str]]:
    real = {metric: np.array([row[metric] for row in stages["real"]]) for metric in METRICS}
    table = []
    for stage, rows in stages.items():
        for metric in METRICS:
            values = np.array([row[metric] for row in rows])
            ratio, low, high = paired_mean_ratio(values, real[metric], RESAMPLES, SEED)
            table.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ratio_to_real": ratio,
                    "ratio_ci_lo": low,
                    "ratio_ci_hi": high,
                    "clips": len(rows),
                }
            )
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=256)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/outputs/qa/motion_sharpness_val.json")
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("docs/paper/pics/motion_sharpness_validation.csv")
    )
    arguments = parser.parse_args()

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    tokenizer = app.models.load_tokenizer(TOKENIZER)
    resources = app.create_humanml3d_generation_evaluator().resources
    downsample = app.config.tokenizer.downsample
    clips = validation_clips(resources, downsample, arguments.clips)
    stages = {
        "real": [sharpness_statistics(recover_skeleton(raw)) for _, raw, _ in clips],
        "tokenizer_roundtrip": roundtrip_rows(tokenizer, resources, clips, app.device),
    }
    for backbone, checkpoint in GENERATORS:
        pipeline = app.models.load_text_to_motion_generator(
            backbone, checkpoint, tokenizer, tokenizer_checkpoint=TOKENIZER
        ).eval()
        stages[f"generated_{backbone.value}"] = generated_rows(
            pipeline, resources, clips, downsample
        )
        del pipeline
        torch.cuda.empty_cache()

    table = summary(stages)
    for row in table:
        print(json.dumps(row), flush=True)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(
            {"split": Split.VALIDATION.value, "clip_ids": [c[0] for c in clips], "table": table},
            indent=1,
        ),
        encoding="utf-8",
    )
    arguments.csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
