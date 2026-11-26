from __future__ import annotations

import logging
from pathlib import Path

import torch

from text2motion.common.engine import CleanCrossEntropyProgress, restore_clean_ce
from text2motion.generation.requests import GeneratorTrainingRequest
from text2motion.generation.training.selection import (
    SelectionState,
    check_ce_stop_setup,
    early_stop_reason,
)
from text2motion.generation.training.trainer import (
    GENERATOR_RESUME_FORMAT,
    CleanCrossEntropyLoaders,
    GeneratorTrainer,
)
from text2motion.motion.storage import is_legacy_resume

logger = logging.getLogger(__name__)


def free_cuda_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()


def start_selection(
    trainer: GeneratorTrainer,
    request: GeneratorTrainingRequest,
    device: str,
    resume_path: Path,
    ce_loaders: CleanCrossEntropyLoaders | None,
) -> tuple[int, SelectionState]:
    check_ce_stop_setup(request, ce_loaders)
    if not request.resume:
        progress = CleanCrossEntropyProgress(best=float("inf"), stale=0)
        return 0, SelectionState(float("inf"), 0, progress)
    return restore_selection(trainer, request, device, resume_path)


def restore_selection(
    trainer: GeneratorTrainer, request: GeneratorTrainingRequest, device: str, resume_path: Path
) -> tuple[int, SelectionState]:
    if not resume_path.is_file():
        raise FileNotFoundError(
            f"--resume continues from {resume_path}, which does not exist; "
            "start without --resume, or copy the source run's *_last.pt there first"
        )
    start_epoch, selection = read_resume_selection(trainer, request, resume_path)
    if request.fresh_selection:
        reset_fresh_selection(selection)
    free_cuda_memory(device)
    logger.info(
        f"resumed from {resume_path} at epoch {start_epoch} (best FID {selection.best_fid:.4f})"
    )
    return start_epoch, selection


def read_resume_selection(
    trainer: GeneratorTrainer, request: GeneratorTrainingRequest, resume_path: Path
) -> tuple[int, SelectionState]:
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
    del state
    return start_epoch, SelectionState(best_fid, stale_evaluations, ce_progress)


def reset_fresh_selection(selection: SelectionState) -> None:
    logger.info(
        f"fresh selection: ignoring inherited best FID {selection.best_fid:.4f} "
        "and resetting validation-CE patience"
    )
    selection.best_fid = float("inf")
    selection.stale_evaluations = 0


def resume_is_finished(
    request: GeneratorTrainingRequest, start_epoch: int, selection: SelectionState
) -> bool:
    reason = early_stop_reason(
        request,
        start_epoch - 1,
        selection.stale_evaluations,
        selection.best_fid,
        selection.ce_progress,
    )
    if reason:
        logger.info(reason)
    return bool(reason)
