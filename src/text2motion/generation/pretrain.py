from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from text2motion.generation.contracts import (
    GeneratorPretrainingRequest,
    GeneratorTrainingConfig,
    MixedPrecisionMode,
)
from text2motion.generation.datasets import build_token_loaders
from text2motion.generation.losses import token_ce_loss
from text2motion.generation.nn_model import GeneratorModelSpec
from text2motion.generation.trainer import (
    adamw_groups,
    check_schedule_horizon,
    warmup_cosine_factor,
)
from text2motion.motion.contracts import MetricSink
from text2motion.motion.datasets import dataset_size
from text2motion.motion.storage import (
    RESUME_VERSION,
    atomic_torch_save,
    capture_rng_state,
    is_legacy_resume,
    restore_rng_state,
    write_failure_dump,
)

logger = logging.getLogger(__name__)


PRETRAIN_RESUME_FORMAT = "text2motion.pretrain_resume"


def pretraining_resume_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_last.pt")


def __load_pretraining_resume(
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


def __pretraining_state(
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


def pretrain_generator(
    architecture: GeneratorModelSpec,
    generator,
    request: GeneratorPretrainingRequest,
    train: GeneratorTrainingConfig,
    device: str,
    out_path: Path,
    on_metrics: MetricSink = lambda metrics: None,
    seed: int = 2026,
    run_dir: Path | None = None,
    failure_dump_path: Path | None = None,
) -> None:
    loader, val_loader, budget = build_token_loaders(request, seed)
    logger.info(
        f"pretrain split by clip: {dataset_size(loader)} train windows, "
        f"{0 if val_loader is None else dataset_size(val_loader)} val windows"
    )

    groups = adamw_groups(generator, train.lr, train.decay_groups)
    opt = torch.optim.AdamW(groups, lr=train.lr, weight_decay=train.weight_decay)
    total_steps = request.epochs * budget.optimizer_steps
    warmup = train.warmup_steps
    floor = train.lr_min_ratio

    def lr_multiplier(step: int) -> float:
        return warmup_cosine_factor(step, warmup, total_steps, floor)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = pretraining_resume_path(out_path)

    amp_on = train.amp == MixedPrecisionMode.BF16 and device == "cuda"
    logger.info(
        f"segments {dataset_size(loader)}  examples/epoch {budget.consumed_examples}  "
        f"microbatches/epoch {budget.microbatches}  "
        f"optimizer_steps/epoch {budget.optimizer_steps}  "
        f"effective_batch {budget.effective_batch}  device {device}  amp {train.amp}"
    )

    step = 0
    start_epoch = 0
    if request.resume and not resume_path.is_file():
        raise FileNotFoundError(
            f"--resume continues from {resume_path}, which does not exist; start without --resume"
        )
    if request.resume:
        step, start_epoch = __load_pretraining_resume(
            generator, opt, resume_path, request, total_steps, device
        )

    def null_condition(batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            architecture.config.text_prefix_len,
            architecture.config.d_text,
            device=device,
        )

    for epoch in range(start_epoch, request.epochs):
        generator.train()
        running = 0.0
        seen = 0
        opt.zero_grad(set_to_none=True)
        inputs: dict[str, torch.Tensor] = {}

        try:
            for microbatch_index, (tokens, lengths) in enumerate(loader):
                if microbatch_index >= budget.microbatches:
                    break
                tokens, lengths = tokens.to(device), lengths.to(device)
                inputs = {"tokens": tokens, "lengths": lengths}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                    raw_loss = token_ce_loss(
                        generator(tokens, null_condition(tokens.size(0))), tokens, lengths
                    )
                    loss = raw_loss / request.grad_accum
                loss.backward()
                if (microbatch_index + 1) % request.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        generator.parameters(), train.grad_clip, error_if_nonfinite=True
                    )
                    for group in opt.param_groups:
                        group["lr"] = train.lr * lr_multiplier(step)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                running += raw_loss.item()
                seen += 1
        except Exception as error:
            write_failure_dump(
                failure_dump_path,
                __pretraining_state(generator, opt, total_steps, run_dir, step, epoch),
                inputs,
                seen,
                generator,
                error,
            )
            raise

        if step != (epoch + 1) * budget.optimizer_steps:
            raise RuntimeError(
                "pretraining optimizer-step budget drifted: "
                f"got {step}, expected {(epoch + 1) * budget.optimizer_steps}"
            )

        record = {"epoch": epoch + 1, "ce": running / seen}
        line = f"pretrain ep {epoch + 1:3d}  ce {running / seen:.4f}"

        if val_loader is not None:
            generator.eval()
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
            record["val_ce"] = val_total / val_seen
            record["gap"] = record["val_ce"] - record["ce"]
            line += f"  val_ce {record['val_ce']:.4f}  gap {record['gap']:+.4f}"

        on_metrics(record)
        logger.info(line)
        atomic_torch_save(generator.state_dict(), out_path)
        atomic_torch_save(
            __pretraining_state(generator, opt, total_steps, run_dir, step, epoch), resume_path
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path.with_suffix(out_path.suffix + ".done").write_text(
        f"epochs {request.epochs}", encoding="utf-8"
    )
    logger.info(f"saved pretrained init -> {out_path}")
