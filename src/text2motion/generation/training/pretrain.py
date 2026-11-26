from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import torch

from text2motion.common.engine import run_epochs
from text2motion.dataset.training.batching import dataset_size
from text2motion.generation.contracts import GeneratorTrainingConfig
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.requests import GeneratorPretrainingRequest
from text2motion.generation.token_datasets import build_token_loaders
from text2motion.generation.training.optimizer import adamw_groups, check_schedule_horizon
from text2motion.generation.training.pretrain_epoch import (
    PRETRAIN_RESUME_FORMAT,
    PretrainEpochContext,
    PretrainProgress,
    after_pretrain_epoch,
    pretrain_guard,
    train_pretrain_epoch,
)
from text2motion.motion.contracts import NO_RUN_OUTPUTS, RunOutputs
from text2motion.motion.storage import is_legacy_resume, restore_rng_state

logger = logging.getLogger(__name__)


def pretraining_resume_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_last.pt")


def load_pretraining_resume(
    generator,
    optimizer: torch.optim.Optimizer,
    resume_path: Path,
    request: GeneratorPretrainingRequest,
    total_steps: int,
    device: str,
) -> tuple[int, int]:
    state = torch.load(resume_path, map_location=device, weights_only=True)
    legacy = is_legacy_resume(
        state, resume_path, PRETRAIN_RESUME_FORMAT, request.accept_legacy_resume
    )
    generator.load_state_dict(state["generator"])
    optimizer.load_state_dict(state["optimizer"])
    step, start_epoch = state["step"], state["epoch"] + 1
    logger.info(f"resumed pretrain from {resume_path} at epoch {start_epoch} (step {step})")
    if legacy:
        logger.warning(
            "old resume file: the RNG state and the schedule horizon are not in it; they start fresh"
        )
        return step, start_epoch
    check_schedule_horizon(state["total_steps"], total_steps, request.extend_schedule)
    restore_rng_state(state["rng"])
    return step, start_epoch


def build_pretraining_optimizer(generator, train: GeneratorTrainingConfig) -> torch.optim.AdamW:
    groups = adamw_groups(generator, train.lr, train.decay_groups)
    return torch.optim.AdamW(groups, lr=train.lr, weight_decay=train.weight_decay)


def prepare_pretrain_output(out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def log_pretrain_setup(loader, val_loader, budget, device: str, train: GeneratorTrainingConfig):
    logger.info(
        f"pretrain split by clip: {dataset_size(loader)} train windows, "
        f"{0 if val_loader is None else dataset_size(val_loader)} val windows"
    )
    logger.info(
        f"segments {dataset_size(loader)}  examples/epoch {budget.consumed_examples}  "
        f"microbatches/epoch {budget.microbatches}  "
        f"optimizer_steps/epoch {budget.optimizer_steps}  "
        f"effective_batch {budget.effective_batch}  device {device}  amp {train.amp}"
    )


def build_pretrain_context(
    architecture: GeneratorModelSpec,
    generator,
    request: GeneratorPretrainingRequest,
    train: GeneratorTrainingConfig,
    device: str,
    out_path: Path,
    outputs: RunOutputs,
    seed: int,
) -> PretrainEpochContext:
    loader, val_loader, budget = build_token_loaders(request, seed)
    log_pretrain_setup(loader, val_loader, budget, device, train)
    optimizer = build_pretraining_optimizer(generator, train)
    out_path = prepare_pretrain_output(out_path)
    return PretrainEpochContext(
        architecture=architecture,
        generator=generator,
        optimizer=optimizer,
        loader=loader,
        val_loader=val_loader,
        budget=budget,
        request=request,
        train=train,
        device=device,
        outputs=outputs,
        out_path=out_path,
        resume_path=pretraining_resume_path(out_path),
        progress=PretrainProgress(step=0),
    )


def resume_pretraining(ctx: PretrainEpochContext) -> tuple[int, int]:
    if not ctx.request.resume:
        return 0, 0
    if not ctx.resume_path.is_file():
        raise FileNotFoundError(
            f"--resume continues from {ctx.resume_path}, which does not exist; start without --resume"
        )
    return load_pretraining_resume(
        ctx.generator, ctx.optimizer, ctx.resume_path, ctx.request, ctx.total_steps, ctx.device
    )


def run_pretrain_epochs(ctx: PretrainEpochContext) -> None:
    step, start_epoch = resume_pretraining(ctx)
    ctx.progress.step = step
    run_epochs(
        start_epoch,
        ctx.request.epochs,
        partial(train_pretrain_epoch, ctx),
        partial(after_pretrain_epoch, ctx),
        partial(pretrain_guard, ctx),
    )


def finish_pretraining(out_path: Path, epochs: int) -> None:
    out_path.with_suffix(out_path.suffix + ".done").write_text(f"epochs {epochs}", encoding="utf-8")
    logger.info(f"saved pretrained init -> {out_path}")


def pretrain_generator(
    architecture: GeneratorModelSpec,
    generator,
    request: GeneratorPretrainingRequest,
    train: GeneratorTrainingConfig,
    device: str,
    out_path: Path,
    outputs: RunOutputs = NO_RUN_OUTPUTS,
    seed: int = 2026,
) -> None:
    ctx = build_pretrain_context(
        architecture, generator, request, train, device, out_path, outputs, seed
    )
    run_pretrain_epochs(ctx)
    finish_pretraining(ctx.out_path, request.epochs)
