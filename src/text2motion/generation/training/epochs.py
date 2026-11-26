from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

from text2motion.common.engine import run_epochs
from text2motion.generation.requests import GeneratorTrainingRequest
from text2motion.generation.training.resume import (
    free_cuda_memory,
    resume_is_finished,
    start_selection,
)
from text2motion.generation.training.selection import (
    SelectionState,
    advance_clean_ce,
    early_stop_reason,
    is_eligible_improvement,
    save_on_lower_fid,
    save_on_lower_val_ce,
)
from text2motion.generation.training.trainer import (
    CheckpointSink,
    CleanCrossEntropyLoaders,
    GeneratorTrainer,
)
from text2motion.motion.contracts import NO_RUN_OUTPUTS, Evaluator, RunOutputs, Split
from text2motion.motion.storage import atomic_torch_save, dump_on_failure

logger = logging.getLogger(__name__)


def finish_epoch(
    trainer: GeneratorTrainer,
    request: GeneratorTrainingRequest,
    device: str,
    evaluate: Evaluator,
    save_best: CheckpointSink,
    resume_path: Path,
    outputs: RunOutputs,
    ce_loaders: CleanCrossEntropyLoaders | None,
    selection: SelectionState,
    epoch: int,
    means: dict[str, float],
) -> str:
    report_epoch(trainer, request, outputs, selection, epoch, means)
    update_clean_ce(trainer, request, device, ce_loaders, outputs, selection, save_best, epoch)
    if validation_due(request, epoch):
        evaluate_validation(trainer, request, evaluate, save_best, outputs, selection, epoch)
    save_resume_state(trainer, selection, epoch, outputs, resume_path)
    free_cuda_memory(device)
    return early_stop_reason(
        request, epoch, selection.stale_evaluations, selection.best_fid, selection.ce_progress
    )


def report_epoch(
    trainer: GeneratorTrainer,
    request: GeneratorTrainingRequest,
    outputs: RunOutputs,
    selection: SelectionState,
    epoch: int,
    means: dict[str, float],
) -> None:
    epoch_minutes = (time.perf_counter() - selection.epoch_started) / 60
    learning_rate = trainer.optimizer.param_groups[0]["lr"]
    outputs.on_metrics(
        {"epoch": epoch + 1, **means, "lr": learning_rate, "epoch_minutes": epoch_minutes}
    )
    logger.info(
        f"epoch {epoch + 1:3d}/{request.epochs}  ce {means['ce']:.4f}  "
        f"ric {means['ric']:.4f}  rot6d {means['rot6d']:.4f}  foot {means['foot']:.4f}  "
        f"total {means['total']:.4f}  lr {learning_rate:.2e}  {epoch_minutes:.1f} min"
    )
    logger.debug(f"epoch {epoch + 1} all loss terms {means}")


def update_clean_ce(
    trainer: GeneratorTrainer,
    request: GeneratorTrainingRequest,
    device: str,
    ce_loaders: CleanCrossEntropyLoaders | None,
    outputs: RunOutputs,
    selection: SelectionState,
    save_best: CheckpointSink,
    epoch: int,
) -> None:
    selection.ce_progress = advance_clean_ce(
        trainer,
        ce_loaders,
        device,
        request.ce_batches,
        epoch,
        outputs.on_metrics,
        selection.ce_progress,
    )
    save_on_lower_val_ce(trainer, request, selection.ce_progress, epoch, save_best)


def validation_due(request: GeneratorTrainingRequest, epoch: int) -> bool:
    return (epoch + 1) % request.eval_every == 0 or epoch + 1 in (
        request.epochs,
        request.stop_epoch,
    )


def save_resume_state(
    trainer: GeneratorTrainer,
    selection: SelectionState,
    epoch: int,
    outputs: RunOutputs,
    resume_path: Path,
) -> None:
    state = trainer.resume_state(
        epoch, selection.best_fid, selection.stale_evaluations, outputs.run_dir
    )
    state["clean_ce_progress"] = {
        "best": selection.ce_progress.best,
        "stale": selection.ce_progress.stale,
    }
    atomic_torch_save(state, resume_path)
    del state


def evaluate_validation(
    trainer: GeneratorTrainer,
    request: GeneratorTrainingRequest,
    evaluate: Evaluator,
    save_best: CheckpointSink,
    outputs: RunOutputs,
    selection: SelectionState,
    epoch: int,
) -> None:
    with trainer.ema_weights():
        metrics = evaluate(Split.VALIDATION)
    outputs.on_metrics({"epoch": epoch + 1, "split": Split.VALIDATION.value, **metrics})
    logger.info(
        f"  [gen-eval:{Split.VALIDATION.value}] clips {metrics['clips']}  "
        f"FID {metrics['fid']:.4f}  R@1 {metrics['r_top1']:.3f}  "
        f"R@3 {metrics['r_top3']:.3f}  MM {metrics['mm_dist']:.3f}  "
        f"Div {metrics['diversity']:.3f}"
    )
    not_finite = sorted(name for name, value in metrics.items() if not math.isfinite(value))
    if not_finite:
        raise RuntimeError(f"epoch {epoch + 1}: evaluation metrics not finite: {not_finite}")
    if is_eligible_improvement(metrics, selection.best_fid, request):
        selection.best_fid = metrics["fid"]
        selection.stale_evaluations = 0
        save_on_lower_fid(trainer, request, epoch, selection.best_fid, save_best)
    else:
        selection.stale_evaluations += 1


def train_generator(
    trainer: GeneratorTrainer,
    loader,
    request: GeneratorTrainingRequest,
    device: str,
    evaluate: Evaluator,
    save_best: CheckpointSink,
    resume_path: Path,
    outputs: RunOutputs = NO_RUN_OUTPUTS,
    ce_loaders: CleanCrossEntropyLoaders | None = None,
) -> float:
    start_epoch, selection = start_selection(trainer, request, device, resume_path, ce_loaders)
    if request.resume and resume_is_finished(request, start_epoch, selection):
        return selection.best_fid
    after_epoch = make_after_epoch(
        trainer, request, device, evaluate, save_best, resume_path, outputs, ce_loaders, selection
    )
    run_epochs(
        start_epoch,
        request.epochs,
        lambda epoch: run_training_epoch(trainer, loader, device, outputs, selection, epoch),
        after_epoch,
        lambda epoch: epoch_guard(trainer, outputs, selection, epoch),
    )
    return selection.best_fid


def make_after_epoch(
    trainer: GeneratorTrainer,
    request: GeneratorTrainingRequest,
    device: str,
    evaluate: Evaluator,
    save_best: CheckpointSink,
    resume_path: Path,
    outputs: RunOutputs,
    ce_loaders: CleanCrossEntropyLoaders | None,
    selection: SelectionState,
) -> Callable[[int, dict[str, float]], str]:
    return lambda epoch, means: finish_epoch(
        trainer,
        request,
        device,
        evaluate,
        save_best,
        resume_path,
        outputs,
        ce_loaders,
        selection,
        epoch,
        means,
    )


def run_training_epoch(
    trainer: GeneratorTrainer,
    loader,
    device: str,
    outputs: RunOutputs,
    selection: SelectionState,
    epoch: int,
) -> dict[str, float]:
    heartbeat = None if outputs.run_dir is None else Path(outputs.run_dir) / "heartbeat"
    loader.dataset.set_epoch(epoch)
    selection.epoch_started = time.perf_counter()
    return trainer.train_epoch(loader, device, heartbeat=heartbeat)


def epoch_guard(
    trainer: GeneratorTrainer, outputs: RunOutputs, selection: SelectionState, epoch: int
):
    return dump_on_failure(
        outputs.failure_dump_path,
        lambda: (
            trainer.resume_state(
                epoch, selection.best_fid, selection.stale_evaluations, outputs.run_dir
            ),
            trainer.current_inputs,
            trainer.current_step,
            trainer.trained_modules(),
        ),
    )
