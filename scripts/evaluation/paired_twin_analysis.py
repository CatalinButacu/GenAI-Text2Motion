# .\.venv\Scripts\python.exe scripts/evaluation/paired_twin_analysis.py --split val
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from text2motion.app.runtime.context import ApplicationBootstrap
from text2motion.app.runtime.run_log import log_metrics, start_run
from text2motion.evaluation.contracts import GenerationEvaluationSet
from text2motion.evaluation.embeddings import compute_guo_text_embeddings
from text2motion.evaluation.metrics import (
    fid,
    motion_statistics,
    own_caption_ranks,
    paired_fid_delta,
)
from text2motion.generation.contracts import Backbone, StudioSamplingConfig
from text2motion.motion.contracts import Split
from text2motion.motion.recovery import recover_skeleton

logger = logging.getLogger(__name__)
SEED = 2026
PAPER_PICS = Path("docs/paper/pics")
PLAUSIBILITY_KEYS = (
    "foot_skate_m_per_s",
    "mean_acceleration_cm_per_frame2",
    "mean_jerk_cm_per_frame3",
)


def paired_summary(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    difference = first - second
    wilcoxon_p = float(np.asarray(stats.wilcoxon(first, second))[1])
    second_lower = int((difference > 0).sum())
    first_lower = int((difference < 0).sum())
    sign = stats.binomtest(second_lower, second_lower + first_lower, 0.5)
    return {
        "mean_first": float(first.mean()),
        "mean_second": float(second.mean()),
        "mean_difference": float(difference.mean()),
        "second_lower_count": second_lower,
        "first_lower_count": first_lower,
        "sign_test_p": float(sign.pvalue),
        "wilcoxon_p": wilcoxon_p,
    }


def rank_one_interval(hits: np.ndarray, resamples: int) -> dict[str, float]:
    result = stats.bootstrap(
        (hits.astype(np.float64),),
        np.mean,
        n_resamples=resamples,
        method="percentile",
        rng=np.random.default_rng(SEED),
    )
    return {
        "r1": float(hits.mean()),
        "r1_ci_lo": float(result.confidence_interval.low),
        "r1_ci_hi": float(result.confidence_interval.high),
    }


def write_paper_table(path: Path, report: dict) -> None:
    fid_rows = report["fid"]
    rank_rows = report["rank_one_interval"]
    difference = report["fid_difference_transformer_minus_mamba"]
    rank_one = report["rank_one"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "transformer", "mamba", "difference_or_test"])
        writer.writerow(
            [
                "fid_run0_paired_diff_ci95",
                f"{fid_rows['transformer']['fid']:.4f}",
                f"{fid_rows['mamba']['fid']:.4f}",
                f"T-M {difference['delta_mean']:+.4f} "
                f"[{difference['delta_ci_lo']:+.4f},{difference['delta_ci_hi']:+.4f}]",
            ]
        )
        writer.writerow(
            [
                "r1_ci95",
                f"{rank_rows['transformer']['r1']:.4f} [{rank_rows['transformer']['r1_ci_lo']:.4f},"
                f"{rank_rows['transformer']['r1_ci_hi']:.4f}]",
                f"{rank_rows['mamba']['r1']:.4f} [{rank_rows['mamba']['r1_ci_lo']:.4f},"
                f"{rank_rows['mamba']['r1_ci_hi']:.4f}]",
                f"McNemar p {rank_one['mcnemar_exact_p']:.3g}",
            ]
        )
        writer.writerow(["clips", report["clips"], report["clips"], report["split"]])


def plausibility_table(features: list[np.ndarray]) -> dict[str, np.ndarray]:
    rows = [motion_statistics(recover_skeleton(feature)) for feature in features]
    return {key: np.array([row[key] for row in rows]) for key in PLAUSIBILITY_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/a7/width_320_step50.yaml")
    parser.add_argument(
        "--tokenizer_ckpt",
        type=Path,
        default=Path("runs/checkpoints/tokenizer/fsq_g8_v1024.pt"),
    )
    parser.add_argument("--checkpoint_pattern", default="generator_{backbone}_w320_a7_step50.pt")
    parser.add_argument("--split", type=Split, default=Split.VALIDATION)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--max_clips", type=int, default=None)
    args = parser.parse_args()

    app = ApplicationBootstrap.from_config(args.config, seed=SEED)
    tokenizer = app.models.load_tokenizer(args.tokenizer_ckpt)
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    run_dir = start_run("paired_twin_analysis", app.config, app.config.paths.logs_dir, vars(args))
    sampling = StudioSamplingConfig()

    sets: dict[Backbone, GenerationEvaluationSet] = {}
    for backbone in (Backbone.TRANSFORMER, Backbone.MAMBA):
        checkpoint = app.checkpoint_directory("generator") / args.checkpoint_pattern.format(
            backbone=backbone.value
        )
        generator = app.models.load_text_to_motion_generator(
            backbone, checkpoint, tokenizer, tokenizer_checkpoint=args.tokenizer_ckpt
        )
        sets[backbone] = evaluator.generate_evaluation_set(
            generator, args.split, args.max_clips, sampling, batch_size=32
        )
        logger.info(f"{backbone.value}: generated {len(sets[backbone].generated)} clips")
        del generator
        torch.cuda.empty_cache()

    transformer_set = sets[Backbone.TRANSFORMER]
    mamba_set = sets[Backbone.MAMBA]
    if transformer_set.captions != mamba_set.captions:
        raise RuntimeError("the two twins did not generate the same clips in the same order")

    reference = evaluator.compute_motion_embeddings(transformer_set.reference)
    transformer_motion = evaluator.compute_motion_embeddings(transformer_set.generated)
    mamba_motion = evaluator.compute_motion_embeddings(mamba_set.generated)
    first_captions = [
        resources.build_guo_text_inputs(tokens[0]) for tokens in transformer_set.token_lists
    ]
    text = compute_guo_text_embeddings(resources.text_embedder, first_captions, resources.device)
    clip_count = len(text)

    fid_difference = paired_fid_delta(
        reference, transformer_motion, mamba_motion, resamples=args.resamples, seed=SEED
    )

    transformer_distance = np.linalg.norm(text - transformer_motion, axis=-1)
    mamba_distance = np.linalg.norm(text - mamba_motion, axis=-1)
    text_distance = paired_summary(transformer_distance, mamba_distance)

    order = np.random.default_rng(SEED).permutation(clip_count)
    transformer_rank = own_caption_ranks(text, transformer_motion, order)
    mamba_rank = own_caption_ranks(text, mamba_motion, order)
    pooled = transformer_rank >= 0
    transformer_hit = transformer_rank[pooled] == 0
    mamba_hit = mamba_rank[pooled] == 0
    only_mamba = int((mamba_hit & ~transformer_hit).sum())
    only_transformer = int((transformer_hit & ~mamba_hit).sum())
    rank_one = {
        "clips_in_pools": int(pooled.sum()),
        "transformer_rate": float(transformer_hit.mean()),
        "mamba_rate": float(mamba_hit.mean()),
        "only_mamba_correct": only_mamba,
        "only_transformer_correct": only_transformer,
        "mcnemar_exact_p": float(
            stats.binomtest(only_mamba, only_mamba + only_transformer, 0.5).pvalue
        ),
    }

    real_plausibility = plausibility_table(transformer_set.reference)
    transformer_plausibility = plausibility_table(transformer_set.generated)
    mamba_plausibility = plausibility_table(mamba_set.generated)
    plausibility = {
        key: {
            "real_mean": float(real_plausibility[key].mean()),
            **paired_summary(transformer_plausibility[key], mamba_plausibility[key]),
        }
        for key in PLAUSIBILITY_KEYS
    }

    fid_rows = {
        "transformer": {"fid": fid(reference, transformer_motion)},
        "mamba": {"fid": fid(reference, mamba_motion)},
    }
    rank_rows = {
        "transformer": rank_one_interval(transformer_hit, args.resamples),
        "mamba": rank_one_interval(mamba_hit, args.resamples),
    }

    report = {
        "split": args.split.value,
        "clips": clip_count,
        "fid": fid_rows,
        "rank_one_interval": rank_rows,
        "first_model": "transformer",
        "second_model": "mamba",
        "fid_difference_transformer_minus_mamba": fid_difference,
        "text_distance": text_distance,
        "rank_one": rank_one,
        "plausibility": plausibility,
    }
    log_metrics(run_dir, report)
    out_path = app.config.paths.outputs_dir / f"paired_twin_analysis_{args.split.value}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(f"clips {clip_count}")
    logger.info(
        f"E1 FID difference (transformer - mamba): mean {fid_difference['delta_mean']:+.4f}  "
        f"95% [{fid_difference['delta_ci_lo']:+.4f}, {fid_difference['delta_ci_hi']:+.4f}]  "
        f"mamba lower in {fid_difference['share_a_higher']:.1%} of {args.resamples} resamples"
    )
    logger.info(
        f"E2 text distance: transformer {text_distance['mean_first']:.4f}  "
        f"mamba {text_distance['mean_second']:.4f}  mamba nearer on "
        f"{text_distance['second_lower_count']} clips, transformer on "
        f"{text_distance['first_lower_count']}  sign p {text_distance['sign_test_p']:.2e}  "
        f"Wilcoxon p {text_distance['wilcoxon_p']:.2e}"
    )
    logger.info(
        f"E2 rank 1: transformer {rank_one['transformer_rate']:.4f}  mamba "
        f"{rank_one['mamba_rate']:.4f}  only mamba correct {only_mamba}  only transformer "
        f"correct {only_transformer}  McNemar p {rank_one['mcnemar_exact_p']:.2e}"
    )
    for key, row in plausibility.items():
        logger.info(
            f"E3 {key}: real {row['real_mean']:.4f}  transformer {row['mean_first']:.4f}  "
            f"mamba {row['mean_second']:.4f}  Wilcoxon p {row['wilcoxon_p']:.2e}"
        )
    table = PAPER_PICS / f"a9_f2_paired_twin_{args.split.value}.csv"
    write_paper_table(table, report)
    logger.info(f"wrote {out_path} and {table}")


if __name__ == "__main__":
    main()
