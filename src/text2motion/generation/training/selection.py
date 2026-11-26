from __future__ import annotations

import logging
from dataclasses import dataclass

from text2motion.common.engine import CleanCrossEntropyProgress
from text2motion.generation.requests import GeneratorTrainingRequest
from text2motion.generation.training.trainer import (
    CheckpointSink,
    CleanCrossEntropyLoaders,
    GeneratorTrainer,
)
from text2motion.motion.contracts import MetricSink

logger = logging.getLogger(__name__)


############################################################
#       SelectionState
############################################################
@dataclass
class SelectionState:
    best_fid: float
    stale_evaluations: int
    ce_progress: CleanCrossEntropyProgress
    epoch_started: float = 0.0


def save_on_lower_val_ce(
    trainer: GeneratorTrainer,
    request: GeneratorTrainingRequest,
    progress: CleanCrossEntropyProgress,
    epoch: int,
    save_best: CheckpointSink,
) -> None:
    if not request.select_on_val_ce or progress.stale:
        return
    with trainer.ema_weights():
        save_best(epoch + 1, float("nan"))
    logger.info(f"  selected on clean validation CE {progress.best:.4f}")


def is_eligible_improvement(
    metrics: dict[str, float], best_fid: float, request: GeneratorTrainingRequest
) -> bool:
    return metrics["fid"] < best_fid and metrics["r_top1"] >= request.save_r1_floor


def save_on_lower_fid(
    trainer: GeneratorTrainer,
    request: GeneratorTrainingRequest,
    epoch: int,
    fid: float,
    save_best: CheckpointSink,
) -> None:
    if request.select_on_val_ce:
        return
    with trainer.ema_weights():
        save_best(epoch + 1, fid)


def check_ce_stop_setup(
    request: GeneratorTrainingRequest, ce_loaders: CleanCrossEntropyLoaders | None
) -> None:
    if request.val_ce_patience < 0:
        raise ValueError("validation-CE patience must be nonnegative")
    if request.val_ce_patience and ce_loaders is None:
        raise ValueError("--val_ce_patience needs the clean CE loaders (keep --clean_ce on)")
    if request.select_on_val_ce and ce_loaders is None:
        raise ValueError("--select_on_val_ce needs the clean CE loaders (keep --clean_ce on)")
    if not 0 <= request.stop_epoch <= request.epochs:
        raise ValueError(f"stop_epoch {request.stop_epoch} out of [0, {request.epochs}]")


def advance_clean_ce(
    trainer: GeneratorTrainer,
    ce_loaders: CleanCrossEntropyLoaders | None,
    device: str,
    max_batches: int,
    epoch: int,
    on_metrics: MetricSink,
    progress: CleanCrossEntropyProgress,
) -> CleanCrossEntropyProgress:
    if ce_loaders is None:
        return progress
    record = clean_ce_record(trainer, ce_loaders, device, max_batches)
    on_metrics({"epoch": epoch + 1, **record})
    logger.info(
        f"  [clean-ce] train {record['train_ce_clean']:.4f}  "
        f"val {record['val_ce']:.4f}  gap {record['ce_gap']:+.4f}"
    )
    return progress.observe(record["val_ce"])


def early_stop_reason(
    request: GeneratorTrainingRequest,
    epoch: int,
    stale_evaluations: int,
    best_fid: float,
    ce_progress: CleanCrossEntropyProgress,
) -> str:
    if request.patience and stale_evaluations >= request.patience:
        return (
            f"early stop at epoch {epoch + 1}: {stale_evaluations} consecutive evaluations "
            f"without improving val FID {best_fid:.4f} (patience {request.patience})"
        )
    if request.stop_epoch and epoch + 1 >= request.stop_epoch:
        return f"planned stop at epoch {epoch + 1} of a {request.epochs}-epoch schedule"
    if request.val_ce_patience and ce_progress.stale >= request.val_ce_patience:
        return (
            f"early stop at epoch {epoch + 1}: {ce_progress.stale} epochs without a lower "
            f"validation CE (best {ce_progress.best:.4f}, patience {request.val_ce_patience})"
        )
    return ""


def clean_ce_record(
    trainer: GeneratorTrainer, ce_loaders: CleanCrossEntropyLoaders, device: str, max_batches: int
) -> dict[str, float]:
    train_ce = trainer.teacher_forced_ce(ce_loaders.train, device, max_batches)
    val_ce = trainer.teacher_forced_ce(ce_loaders.validation, device, max_batches)
    return {"train_ce_clean": train_ce, "val_ce": val_ce, "ce_gap": val_ce - train_ce}
