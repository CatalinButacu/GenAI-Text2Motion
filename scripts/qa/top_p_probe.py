# python scripts/qa/top_p_probe.py --out runs/outputs/qa/top_p_val.json --csv docs/paper/pics/top_p_validation.csv
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.evaluation.metrics import (
    paired_fid_delta,
    paired_mean_ratio,
    sharpness_statistics,
)
from text2motion.evaluation.reports import calculate_run_metrics
from text2motion.generation.contracts import Backbone, SamplingConfig
from text2motion.motion.contracts import Split
from text2motion.motion.representation import recover_skeleton

SEED = 2026
CONFIG = "configs/generator/size_sweep/width_320.yaml"
TOKENIZER = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
GENERATORS = (
    (Backbone.TRANSFORMER, "runs/checkpoints/generator/generator_transformer_w320_p1.pt"),
    (Backbone.MAMBA, "runs/checkpoints/generator/generator_mamba_w320_p1.pt"),
)
BASELINE_TOP_P = 0.9
CHALLENGER_TOP_P = 1.0
MAX_CLIPS = 600
BATCH = 16
RESAMPLES = 1000
METRICS = (
    "peak_speed_m_per_s",
    "peak_acceleration_m_per_s2",
    "mean_jerk_m_per_s3",
    "foot_skate_m_per_s",
)


def sharpness_columns(motions: list[np.ndarray]) -> dict[str, np.ndarray]:
    rows = [sharpness_statistics(recover_skeleton(motion)) for motion in motions]
    return {metric: np.array([row[metric] for row in rows]) for metric in METRICS}


def generated_set(evaluator, pipeline, top_p: float):
    sampling = SamplingConfig(temperature=1.0, top_p=top_p, cfg_scale=6.0)
    return evaluator.generate_evaluation_set(pipeline, Split.VALIDATION, MAX_CLIPS, sampling, BATCH)


def twin_rows(evaluator, pipeline, backbone: Backbone) -> list[dict[str, float | str]]:
    baseline = generated_set(evaluator, pipeline, BASELINE_TOP_P)
    challenger = generated_set(evaluator, pipeline, CHALLENGER_TOP_P)
    if baseline.sample_ids != challenger.sample_ids:
        raise ValueError(f"{backbone.value}: the 2 settings scored different clips")
    real = evaluator.compute_motion_embeddings(baseline.reference)
    caption_emb, per_clip = evaluator.caption_embeddings(baseline)
    rows: list[dict[str, float | str]] = []
    sharp = {"real": sharpness_columns(baseline.reference)}
    embeddings = {}
    for name, generated in (
        (f"top_p {BASELINE_TOP_P}", baseline),
        (f"top_p {CHALLENGER_TOP_P}", challenger),
    ):
        embeddings[name] = evaluator.compute_motion_embeddings(generated.generated)
        metrics = calculate_run_metrics(
            real, embeddings[name], caption_emb, per_clip, np.random.default_rng(0)
        )
        sharp[name] = sharpness_columns(generated.generated)
        rows.append(
            {
                "backbone": backbone.value,
                "setting": name,
                "quantity": "fid",
                "value": metrics["fid"],
            }
        )
        rows.append(
            {
                "backbone": backbone.value,
                "setting": name,
                "quantity": "r_top1",
                "value": metrics["r_top1"],
            }
        )
        rows.append(
            {
                "backbone": backbone.value,
                "setting": name,
                "quantity": "diversity",
                "value": metrics["diversity"],
            }
        )
        for metric in METRICS:
            ratio, low, high = paired_mean_ratio(
                sharp[name][metric], sharp["real"][metric], RESAMPLES, SEED
            )
            rows.append(
                {
                    "backbone": backbone.value,
                    "setting": name,
                    "quantity": f"{metric}_ratio_to_real",
                    "value": ratio,
                    "ci_lo": low,
                    "ci_hi": high,
                }
            )
    challenger_name, baseline_name = f"top_p {CHALLENGER_TOP_P}", f"top_p {BASELINE_TOP_P}"
    delta = paired_fid_delta(
        real, embeddings[challenger_name], embeddings[baseline_name], RESAMPLES, SEED
    )
    rows.append(
        {
            "backbone": backbone.value,
            "setting": f"{challenger_name} minus {baseline_name}",
            "quantity": "fid_delta",
            "value": delta["delta_mean"],
            "ci_lo": delta["delta_ci_lo"],
            "ci_hi": delta["delta_ci_hi"],
        }
    )
    for metric in METRICS:
        ratio, low, high = paired_mean_ratio(
            sharp[challenger_name][metric], sharp[baseline_name][metric], RESAMPLES, SEED
        )
        rows.append(
            {
                "backbone": backbone.value,
                "setting": f"{challenger_name} over {baseline_name}",
                "quantity": f"{metric}_ratio",
                "value": ratio,
                "ci_lo": low,
                "ci_hi": high,
            }
        )
    rows.append(
        {
            "backbone": backbone.value,
            "setting": "both",
            "quantity": "clips",
            "value": len(baseline.sample_ids),
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/outputs/qa/top_p_val.json"))
    parser.add_argument("--csv", type=Path, default=Path("docs/paper/pics/top_p_validation.csv"))
    arguments = parser.parse_args()

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    tokenizer = app.load_tokenizer(TOKENIZER)
    evaluator = app.create_humanml3d_generation_evaluator()
    rows: list[dict[str, float | str]] = []
    for backbone, checkpoint in GENERATORS:
        pipeline = app.load_text_to_motion_generator(
            backbone, checkpoint, tokenizer, tokenizer_checkpoint=TOKENIZER
        ).eval()
        twin = twin_rows(evaluator, pipeline, backbone)
        for row in twin:
            print(json.dumps(row), flush=True)
        rows.extend(twin)
        del pipeline
        torch.cuda.empty_cache()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps({"split": Split.VALIDATION.value, "rows": rows}, indent=1), encoding="utf-8"
    )
    arguments.csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["backbone", "setting", "quantity", "value", "ci_lo", "ci_hi"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
