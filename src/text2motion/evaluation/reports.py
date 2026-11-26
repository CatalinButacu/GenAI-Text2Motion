from __future__ import annotations

import logging

import numpy as np

from text2motion.evaluation.contracts import (
    DIVERSITY_PAIRS,
    R_PRECISION_POOL,
    GenerationEvaluationReport,
    GenerationEvaluationSet,
)
from text2motion.evaluation.metrics import (
    diversity,
    fid,
    mm_dist,
    r_precision,
    replication_summary,
)

logger = logging.getLogger(__name__)


def calculate_run_metrics(
    reference_emb: np.ndarray,
    generated_emb: np.ndarray,
    caption_emb: np.ndarray,
    per_clip: list[np.ndarray],
    rng: np.random.Generator,
) -> dict[str, float]:
    chosen = np.array([rng.choice(indices) for indices in per_clip])
    order = rng.permutation(len(generated_emb))
    selected = caption_emb[chosen][order]
    motions = generated_emb[order]
    top1, top2, top3 = r_precision(selected, motions, pool_size=R_PRECISION_POOL, top_k=3)
    return {
        "fid": fid(reference_emb, generated_emb),
        "r_top1": float(top1),
        "r_top2": float(top2),
        "r_top3": float(top3),
        "mm_dist": mm_dist(selected, motions),
        "diversity": diversity(generated_emb, num_pairs=DIVERSITY_PAIRS),
    }


def build_protocol_report(
    runs: list[dict[str, float]],
    clips: int,
    reps: int,
    bootstrap: dict[str, float],
    multimodality: float,
) -> GenerationEvaluationReport:
    summary = {name: replication_summary([run[name] for run in runs]) for name in runs[0]}
    return GenerationEvaluationReport(
        clips=clips,
        runs=reps,
        fid=summary["fid"][0],
        r_precision=(summary["r_top1"][0], summary["r_top2"][0], summary["r_top3"][0]),
        diversity=summary["diversity"][0],
        multimodality=multimodality,
        matching_distance=summary["mm_dist"][0],
        fid_ci95=summary["fid"][1],
        r_precision_ci95=(summary["r_top1"][1], summary["r_top2"][1], summary["r_top3"][1]),
        diversity_ci95=summary["diversity"][1],
        matching_distance_ci95=summary["mm_dist"][1],
        fid_bootstrap_lo=bootstrap.get("fid_ci_lo", float("nan")),
        fid_bootstrap_hi=bootstrap.get("fid_ci_hi", float("nan")),
    )


def checkpoint_selection_record(
    generated: GenerationEvaluationSet,
    reference_emb: np.ndarray,
    generated_emb: np.ndarray,
    text_emb: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(generated_emb))
    precision = r_precision(
        text_emb[perm], generated_emb[perm], pool_size=R_PRECISION_POOL, top_k=3
    )
    return {
        "clips": len(generated.generated),
        "requested_clips": generated.requested,
        "dropped": sum(generated.dropped.values()),
        "fid": fid(reference_emb, generated_emb),
        "r_top1": float(precision[0]),
        "r_top3": float(precision[2]),
        "mm_dist": mm_dist(text_emb, generated_emb),
        "diversity": diversity(generated_emb),
    }


def log_horizon_table(rows: list[dict[str, float]]) -> None:
    logger.info(f"{'horizon':>8} {'ms/step':>9} {'state':>12} {'cuda peak':>12}")
    for row in rows:
        logger.info(
            f"{row['horizon']:>8} {row['ms_per_step']:>9.2f} "
            f"{row['state_bytes'] / 1e6:>10.2f}MB {row['cuda_peak_bytes'] / 1e6:>10.1f}MB"
        )


def log_concurrency_table(horizon: int, rows: list[dict[str, float]]) -> None:
    logger.info(f"  concurrency @ horizon {horizon}")
    logger.info(f"  {'streams':>8} {'ms/step':>9} {'state':>12} {'per-stream':>12} {'peak':>10}")
    for row in rows:
        logger.info(
            f"  {row['streams']:>8} {row['ms_per_step']:>9.2f} "
            f"{row['state_bytes'] / 1e6:>10.2f}MB "
            f"{row['state_bytes_per_stream'] / 1e6:>10.3f}MB "
            f"{row['cuda_peak_bytes'] / 1e6:>8.1f}MB"
        )
