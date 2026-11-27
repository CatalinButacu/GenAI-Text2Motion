from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from text2motion.common.engine import CleanCrossEntropyProgress, restore_early_stop_state
from text2motion.generation.requests import GeneratorTrainingRequest
from text2motion.generation.training.optimizer import check_schedule_horizon
from text2motion.generation.training.selection import (
    SelectionState,
    check_ce_stop_setup,
    early_stop_reason,
)
from text2motion.generation.training.trainer import (
    CleanCrossEntropyLoaders,
    GeneratorTrainer,
)
from text2motion.motion.storage import (
    RESUME_VERSION,
    capture_rng_state,
    is_legacy_resume,
    restore_rng_state,
)

logger = logging.getLogger(__name__)

GENERATOR_RESUME_FORMAT = "text2motion.generator_resume"


############################################################
#       GeneratorCheckpoint
############################################################
class GeneratorCheckpoint:
    def __init__(self, trainer: GeneratorTrainer) -> None:
        self.trainer = trainer

    def trained_modules(self) -> nn.ModuleDict:
        modules = nn.ModuleDict({"token_generator": self.trainer.token_generator})
        if self.trainer.text_encoder is not None:
            modules["text_encoder"] = self.trainer.text_encoder
        return modules

    def resume_state(
        self, epoch: int, best_fid: float, stale_evaluations: int, run_dir: Path | None
    ) -> dict:
        if self.trainer._accum_count:
            raise RuntimeError("a resume state needs an epoch boundary with no pending gradients")
        trainer = self.trainer
        return {
            "format": GENERATOR_RESUME_FORMAT,
            "version": RESUME_VERSION,
            "generator": trainer.token_generator.state_dict(),
            "text_encoder": trainer.text_encoder.state_dict() if trainer.text_encoder else None,
            "optimizer": trainer.optimizer.state_dict(),
            "ema": trainer.parameter_ema.shadow,
            "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "weighter": trainer.weighter.state_dict() if trainer.weighter else None,
            "total_steps": trainer.total_steps,
            "rng": capture_rng_state(),
            "run_dir": None if run_dir is None else str(run_dir),
            "epoch": epoch,
            "best_fid": best_fid,
            "stale_evaluations": stale_evaluations,
        }

    def load_resume_state(
        self, state: dict, legacy: bool, extend_schedule: bool
    ) -> tuple[int, float, int]:
        trainer = self.trainer
        trainer.token_generator.load_state_dict(state["generator"])
        if trainer.text_encoder is not None and state.get("text_encoder") is not None:
            trainer.text_encoder.load_state_dict(state["text_encoder"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.parameter_ema.load_state_dict(state["ema"])
        if trainer.scheduler is not None and state.get("scheduler") is not None:
            trainer.scheduler.load_state_dict(state["scheduler"])
        if legacy:
            logger.warning(
                "old resume file: the uncertainty weights, the RNG state and the schedule "
                "horizon are not in it; they start fresh"
            )
            return state["epoch"] + 1, state["best_fid"], state.get("stale_evaluations", 0)
        self.load_weighter(state["weighter"])
        check_schedule_horizon(state["total_steps"], trainer.total_steps, extend_schedule)
        restore_rng_state(state["rng"])
        return state["epoch"] + 1, state["best_fid"], state["stale_evaluations"]

    def load_weighter(self, weighter_state: dict | None) -> None:
        weighter = self.trainer.weighter
        if (weighter is None) != (weighter_state is None):
            raise ValueError("the resume file and this run disagree on the uncertainty weighting")
        if weighter is not None and weighter_state is not None:
            weighter.load_state_dict(weighter_state)


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
    start_epoch, best_fid, stale_evaluations = GeneratorCheckpoint(trainer).load_resume_state(
        state, legacy, request.extend_schedule
    )
    ce_progress = restore_early_stop_state(
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
