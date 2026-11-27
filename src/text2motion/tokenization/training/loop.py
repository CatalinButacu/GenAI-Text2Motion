from __future__ import annotations

import logging
import math
from pathlib import Path

import torch

from text2motion.common.engine import run_epochs
from text2motion.motion.contracts import (
    NO_RUN_OUTPUTS,
    Evaluator,
    RunOutputs,
    Split,
)
from text2motion.motion.storage import (
    atomic_torch_save,
    dump_on_failure,
    is_legacy_resume,
)
from text2motion.tokenization.contracts import TokenizerSchedule
from text2motion.tokenization.training.trainer import TOKENIZER_RESUME_FORMAT, TokenizerTrainer

logger = logging.getLogger(__name__)


def resume_start_point(
    trainer: TokenizerTrainer, schedule: TokenizerSchedule, resume_path: Path, device: str
) -> tuple[int, float]:
    if not schedule.resume:
        return 0, float("inf")
    if not resume_path.is_file():
        raise FileNotFoundError(
            f"--resume continues from {resume_path}, which does not exist; start without --resume"
        )
    state = torch.load(resume_path, map_location=device, weights_only=True)
    legacy = is_legacy_resume(
        state, resume_path, TOKENIZER_RESUME_FORMAT, schedule.accept_legacy_resume
    )
    start_epoch, best_fid = trainer.load_resume_state(state, legacy)
    logger.info(
        f"resumed from {resume_path} at epoch {start_epoch} (best recon-FID {best_fid:.4f})"
    )
    return start_epoch, best_fid


def run_training_epoch(
    trainer: TokenizerTrainer, loader, schedule: TokenizerSchedule, device: str, epoch: int
) -> dict[str, float]:
    loader.dataset.set_epoch(epoch)
    return trainer.train_epoch(loader, device, schedule.max_steps)


def failure_dump(trainer: TokenizerTrainer, outputs: RunOutputs, epoch: int, best_fid: float):
    return dump_on_failure(
        outputs.failure_dump_path,
        lambda: (
            trainer.resume_state(epoch, best_fid, outputs.run_dir),
            trainer.current_inputs,
            trainer.current_step,
            trainer.tokenizer_network,
        ),
    )


def report_epoch(outputs: RunOutputs, epoch: int, means: dict[str, float]) -> None:
    outputs.on_metrics({"epoch": epoch + 1, **means})
    logger.info(
        f"epoch {epoch + 1:3d}  [common] recon {means['recon']:.4f} total {means['total']:.4f}  "
        f"[health] perplexity {means['perplexity']:.1f} usage {means['usage_frac']:.1%} "
        f"commit {means['commit']:.4f}"
    )


def budget_reached(schedule: TokenizerSchedule, trainer: TokenizerTrainer) -> bool:
    return schedule.max_steps is not None and trainer.steps_taken >= schedule.max_steps


def evaluation_skipped(epoch: int, schedule: TokenizerSchedule, step_budget_done: bool) -> bool:
    return bool(
        (epoch + 1) % schedule.eval_every and epoch + 1 != schedule.epochs and not step_budget_done
    )


def validate_epoch(
    trainer: TokenizerTrainer, evaluate: Evaluator, outputs: RunOutputs, epoch: int
) -> dict[str, float]:
    with trainer.ema_weights():
        metrics = evaluate(Split.VALIDATION)
    outputs.on_metrics({"epoch": epoch + 1, "split": Split.VALIDATION.value, **metrics})
    logger.info(
        f"  [val] clips {metrics['clips']}  MPJPE {metrics['mpjpe_mm']:.1f}mm  "
        f"feat-L2 {metrics['feature_l2']:.4f}  recon-FID {metrics['recon_fid']:.4f}"
    )

    not_finite = sorted(name for name, value in metrics.items() if not math.isfinite(value))
    if not_finite:
        raise RuntimeError(f"epoch {epoch + 1}: evaluation metrics not finite: {not_finite}")
    return metrics


def keep_best(
    trainer: TokenizerTrainer, checkpoint_path: Path, best_fid: float, metrics: dict[str, float]
) -> float:
    if metrics["recon_fid"] < best_fid:
        best_fid = metrics["recon_fid"]
        with trainer.ema_weights() as module:
            atomic_torch_save(module.state_dict(), checkpoint_path)
        logger.info(f"  saved best (val) -> {checkpoint_path} (val recon-FID {best_fid:.4f})")
    return best_fid


def budget_message(schedule: TokenizerSchedule, trainer: TokenizerTrainer) -> str:
    if budget_reached(schedule, trainer):
        return f"  step budget of {schedule.max_steps} steps reached; the run stops"
    return ""


def train_tokenizer(
    trainer: TokenizerTrainer,
    loader,
    schedule: TokenizerSchedule,
    device: str,
    evaluate: Evaluator,
    checkpoint_path: Path,
    resume_path: Path,
    outputs: RunOutputs = NO_RUN_OUTPUTS,
) -> float:
    start_epoch, best_fid = resume_start_point(trainer, schedule, resume_path, device)

    def after_epoch(epoch: int, means: dict[str, float]) -> str:
        nonlocal best_fid
        report_epoch(outputs, epoch, means)
        if evaluation_skipped(epoch, schedule, budget_reached(schedule, trainer)):
            return ""
        metrics = validate_epoch(trainer, evaluate, outputs, epoch)
        best_fid = keep_best(trainer, checkpoint_path, best_fid, metrics)
        atomic_torch_save(trainer.resume_state(epoch, best_fid, outputs.run_dir), resume_path)
        return budget_message(schedule, trainer)

    run_epochs(
        start_epoch,
        schedule.epochs,
        lambda epoch: run_training_epoch(trainer, loader, schedule, device, epoch),
        after_epoch,
        lambda epoch: failure_dump(trainer, outputs, epoch, best_fid),
    )
    return best_fid
