from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch

from text2motion.common.engine import CleanCrossEntropyProgress, restore_clean_ce, run_epochs
from text2motion.generation.requests import GeneratorTrainingRequest
from text2motion.generation.training.selection import (
    SelectionState,
    advance_clean_ce,
    check_ce_stop_setup,
    early_stop_reason,
    is_eligible_improvement,
    save_on_lower_fid,
    save_on_lower_val_ce,
)
from text2motion.generation.training.trainer import (
    GENERATOR_RESUME_FORMAT,
    CheckpointSink,
    CleanCrossEntropyLoaders,
    GeneratorTrainer,
)
from text2motion.motion.contracts import NO_RUN_OUTPUTS, Evaluator, RunOutputs, Split
from text2motion.motion.storage import (
    atomic_torch_save,
    dump_on_failure,
    is_legacy_resume,
)

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

    due = (epoch + 1) % request.eval_every == 0 or epoch + 1 in (
        request.epochs,
        request.stop_epoch,
    )
    if due:
        evaluate_validation(trainer, request, evaluate, save_best, outputs, selection, epoch)

    state = trainer.resume_state(
        epoch, selection.best_fid, selection.stale_evaluations, outputs.run_dir
    )
    state["clean_ce_progress"] = {
        "best": selection.ce_progress.best,
        "stale": selection.ce_progress.stale,
    }
    atomic_torch_save(state, resume_path)
    del state

    if device == "cuda":
        torch.cuda.empty_cache()

    return early_stop_reason(
        request, epoch, selection.stale_evaluations, selection.best_fid, selection.ce_progress
    )


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
    best_fid = float("inf")
    start_epoch = 0
    stale_evaluations = 0
    ce_progress = CleanCrossEntropyProgress(best=float("inf"), stale=0)
    heartbeat = None if outputs.run_dir is None else Path(outputs.run_dir) / "heartbeat"
    check_ce_stop_setup(request, ce_loaders)

    if request.resume and not resume_path.is_file():
        raise FileNotFoundError(
            f"--resume continues from {resume_path}, which does not exist; "
            "start without --resume, or copy the source run's *_last.pt there first"
        )
    if request.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        legacy = is_legacy_resume(
            state, resume_path, GENERATOR_RESUME_FORMAT, request.accept_legacy_resume
        )
        start_epoch, best_fid, stale_evaluations = trainer.load_resume_state(
            state, legacy, request.extend_schedule
        )
        ce_progress = restore_clean_ce(
            state, start_epoch, request.fresh_selection, request.val_ce_patience
        )
        if request.fresh_selection:
            logger.info(
                f"fresh selection: ignoring inherited best FID {best_fid:.4f} "
                "and resetting validation-CE patience"
            )
            best_fid = float("inf")
            stale_evaluations = 0
        del state
        if device == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"resumed from {resume_path} at epoch {start_epoch} (best FID {best_fid:.4f})")

        reason = early_stop_reason(
            request, start_epoch - 1, stale_evaluations, best_fid, ce_progress
        )
        if reason:
            logger.info(reason)
            return best_fid

    selection = SelectionState(best_fid, stale_evaluations, ce_progress)

    def guard(epoch: int):
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

    def train_epoch(epoch: int) -> dict[str, float]:
        loader.dataset.set_epoch(epoch)
        selection.epoch_started = time.perf_counter()
        return trainer.train_epoch(loader, device, heartbeat=heartbeat)

    def after_epoch(epoch: int, means: dict[str, float]) -> str:
        return finish_epoch(
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

    run_epochs(start_epoch, request.epochs, train_epoch, after_epoch, guard)
    return selection.best_fid
