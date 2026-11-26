from __future__ import annotations

import logging

import torch
from torch import nn

from text2motion.generation.contracts import (
    DEFAULT_OVERFIT_CRITERIA,
    OverfitCriteria,
)
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.text_encoder import CLIPTextEncoder
from text2motion.generation.training.trainer import GeneratorTrainer
from text2motion.tokenization.nn_tokenizer import TokenizerNetwork

logger = logging.getLogger(__name__)


@torch.no_grad()
def teacher_forced_accuracy(
    generator: MotionTokenGenerator,
    tokenizer: TokenizerNetwork,
    downsample: int,
    motion: torch.Tensor,
    text_embedding: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[float, float]:
    usable = (motion.size(1) // downsample) * downsample
    motion = motion[:, :usable]
    target = tokenizer.encode(motion)
    token_lengths = (lengths.clamp(max=usable) // downsample).clamp(min=1, max=target.size(1))
    logits = generator(target, text_embedding)
    valid = torch.arange(target.size(1), device=target.device)[None, :] < token_lengths[:, None]
    valid = valid[..., None].expand_as(target)
    predictions = logits.argmax(-1)
    accuracy = (predictions[valid] == target[valid]).float().mean()
    ce = nn.functional.cross_entropy(logits[valid], target[valid])
    return float(accuracy), float(ce)


def gate_text_encoder(trainer: GeneratorTrainer) -> CLIPTextEncoder:
    if trainer.text_encoder is None:
        raise ValueError("the overfit gate needs the text encoder of the trainer")
    return trainer.text_encoder


def measure_gate(
    trainer: GeneratorTrainer,
    text_encoder: CLIPTextEncoder,
    motion: torch.Tensor,
    captions: list[str],
    lengths: torch.Tensor,
) -> tuple[float, float]:
    generator = trainer.token_generator
    generator.eval()
    text_encoder.eval()
    return teacher_forced_accuracy(
        generator,
        trainer.tokenizer_network,
        trainer.downsample,
        motion,
        text_encoder(captions),
        lengths,
    )


def restore_gate_training(trainer: GeneratorTrainer, text_encoder: CLIPTextEncoder) -> None:
    trainer.token_generator.train()
    text_encoder.train()


def run_gate_steps(
    trainer: GeneratorTrainer,
    text_encoder: CLIPTextEncoder,
    motion: torch.Tensor,
    captions: list[str],
    lengths: torch.Tensor,
    steps: int,
    criteria: OverfitCriteria,
) -> tuple[float | None, dict[str, float], int]:
    first_total: float | None = None
    parts: dict[str, float] = {}
    step = 0

    for step in range(steps):
        text_embedding = trainer.batch.encode(captions)
        parts = {
            name: float(value)
            for name, value in trainer.train_step(motion, text_embedding, lengths).items()
        }
        first_total = parts["total"] if first_total is None else first_total
        if step % 25 and step + 1 != steps:
            continue
        accuracy, ce = measure_gate(trainer, text_encoder, motion, captions, lengths)
        restore_gate_training(trainer, text_encoder)
        logger.info(
            f"step {step + 1:4d} total {parts['total']:.5f} ce {ce:.5f} token_accuracy {accuracy:.4f}"
        )
        if criteria.accepts(accuracy, ce, parts["total"] / first_total):
            break
    return first_total, parts, step


def check_tokenizer_frozen(trainer: GeneratorTrainer) -> None:
    if any(parameter.grad is not None for parameter in trainer.tokenizer_network.parameters()):
        raise RuntimeError("frozen tokenizer received gradients during overfit gate")


def overfit_single_batch(
    trainer: GeneratorTrainer,
    motion: torch.Tensor,
    lengths: torch.Tensor,
    captions: list[str],
    steps: int,
    criteria: OverfitCriteria = DEFAULT_OVERFIT_CRITERIA,
) -> dict[str, float]:
    text_encoder = gate_text_encoder(trainer)
    first_total, parts, step = run_gate_steps(
        trainer, text_encoder, motion, captions, lengths, steps, criteria
    )
    accuracy, ce = measure_gate(trainer, text_encoder, motion, captions, lengths)
    check_tokenizer_frozen(trainer)
    if first_total is None:
        raise RuntimeError("the overfit gate ran no step")
    return {
        "steps": step + 1,
        "first_total": float(first_total),
        "final_total": parts["total"],
        "token_accuracy": accuracy,
        "ce": ce,
    }
