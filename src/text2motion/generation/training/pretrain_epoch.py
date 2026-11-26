from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from text2motion.generation.contracts import GeneratorTrainingConfig, MixedPrecisionMode
from text2motion.generation.losses import token_ce_loss
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.requests import GeneratorPretrainingRequest
from text2motion.generation.token_datasets import PretrainingBudget
from text2motion.generation.training.optimizer import warmup_cosine_factor
from text2motion.motion.contracts import RunOutputs
from text2motion.motion.storage import (
    RESUME_VERSION,
    atomic_torch_save,
    capture_rng_state,
    dump_on_failure,
)

logger = logging.getLogger(__name__)


PRETRAIN_RESUME_FORMAT = "text2motion.pretrain_resume"


def pretraining_state(
    generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    run_dir: Path | None,
    step: int,
    epoch: int,
) -> dict:
    return {
        "format": PRETRAIN_RESUME_FORMAT,
        "version": RESUME_VERSION,
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "total_steps": total_steps,
        "rng": capture_rng_state(),
        "run_dir": None if run_dir is None else str(run_dir),
        "step": step,
        "epoch": epoch,
    }


############################################################
#       PretrainProgress
############################################################
@dataclass
class PretrainProgress:
    step: int
    seen: int = 0
    inputs: dict[str, torch.Tensor] = field(default_factory=dict)


def validation_ce(generator, val_loader, null_condition, device: str, amp_on: bool) -> float:
    val_total = 0.0
    val_seen = 0
    with torch.no_grad():
        for tokens, lengths in val_loader:
            tokens, lengths = tokens.to(device), lengths.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                val_total += float(
                    token_ce_loss(
                        generator(tokens, null_condition(tokens.size(0))), tokens, lengths
                    )
                )
            val_seen += 1
    return val_total / val_seen


@dataclass
class PretrainEpochContext:
    architecture: GeneratorModelSpec
    generator: nn.Module
    optimizer: torch.optim.Optimizer
    loader: DataLoader
    val_loader: DataLoader | None
    budget: PretrainingBudget
    request: GeneratorPretrainingRequest
    train: GeneratorTrainingConfig
    device: str
    outputs: RunOutputs
    out_path: Path
    resume_path: Path
    progress: PretrainProgress

    @property
    def total_steps(self) -> int:
        return self.request.epochs * self.budget.optimizer_steps

    @property
    def amp_on(self) -> bool:
        return self.train.amp == MixedPrecisionMode.BF16 and self.device == "cuda"

    def null_condition(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.architecture.config.text_prefix_len,
            self.architecture.config.d_text,
            device=self.device,
        )

    def lr_multiplier(self, step: int) -> float:
        return warmup_cosine_factor(
            step, self.train.warmup_steps, self.total_steps, self.train.lr_min_ratio
        )


def train_pretrain_epoch(ctx: PretrainEpochContext, epoch: int) -> dict[str, float]:
    ctx.generator.train()
    running = 0.0
    ctx.progress.seen = 0
    ctx.optimizer.zero_grad(set_to_none=True)
    ctx.progress.inputs = {}
    for microbatch_index, (tokens, lengths) in enumerate(ctx.loader):
        if microbatch_index >= ctx.budget.microbatches:
            break
        tokens, lengths = tokens.to(ctx.device), lengths.to(ctx.device)
        ctx.progress.inputs = {"tokens": tokens, "lengths": lengths}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ctx.amp_on):
            raw_loss = token_ce_loss(
                ctx.generator(tokens, ctx.null_condition(tokens.size(0))), tokens, lengths
            )
            loss = raw_loss / ctx.request.grad_accum
        loss.backward()
        if (microbatch_index + 1) % ctx.request.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                ctx.generator.parameters(), ctx.train.grad_clip, error_if_nonfinite=True
            )
            for group in ctx.optimizer.param_groups:
                group["lr"] = ctx.train.lr * ctx.lr_multiplier(ctx.progress.step)
            ctx.optimizer.step()
            ctx.optimizer.zero_grad(set_to_none=True)
            ctx.progress.step += 1
        running += raw_loss.item()
        ctx.progress.seen += 1
    return {"ce": running / ctx.progress.seen}


def check_pretrain_step_budget(ctx: PretrainEpochContext, epoch: int) -> None:
    if ctx.progress.step != (epoch + 1) * ctx.budget.optimizer_steps:
        raise RuntimeError(
            "pretraining optimizer-step budget drifted: "
            f"got {ctx.progress.step}, expected {(epoch + 1) * ctx.budget.optimizer_steps}"
        )


def after_pretrain_epoch(ctx: PretrainEpochContext, epoch: int, means: dict[str, float]) -> str:
    check_pretrain_step_budget(ctx, epoch)
    record = {"epoch": epoch + 1, "ce": means["ce"]}
    line = f"pretrain ep {epoch + 1:3d}  ce {means['ce']:.4f}"
    if ctx.val_loader is not None:
        ctx.generator.eval()
        record["val_ce"] = validation_ce(
            ctx.generator, ctx.val_loader, ctx.null_condition, ctx.device, ctx.amp_on
        )
        record["gap"] = record["val_ce"] - record["ce"]
        line += f"  val_ce {record['val_ce']:.4f}  gap {record['gap']:+.4f}"
    ctx.outputs.on_metrics(record)
    logger.info(line)
    atomic_torch_save(ctx.generator.state_dict(), ctx.out_path)
    atomic_torch_save(
        pretraining_state(
            ctx.generator,
            ctx.optimizer,
            ctx.total_steps,
            ctx.outputs.run_dir,
            ctx.progress.step,
            epoch,
        ),
        ctx.resume_path,
    )
    if ctx.device == "cuda":
        torch.cuda.empty_cache()
    return ""


def pretrain_guard(ctx: PretrainEpochContext, epoch: int):
    return dump_on_failure(
        ctx.outputs.failure_dump_path,
        lambda: (
            pretraining_state(
                ctx.generator,
                ctx.optimizer,
                ctx.total_steps,
                ctx.outputs.run_dir,
                ctx.progress.step,
                epoch,
            ),
            ctx.progress.inputs,
            ctx.progress.seen,
            ctx.generator,
        ),
    )
