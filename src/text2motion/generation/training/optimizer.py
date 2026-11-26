from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from text2motion.generation.contracts import (
    GeneratorFreeze,
    GeneratorTrainingConfig,
    LossWeighting,
)
from text2motion.generation.losses import UncertaintyWeighter
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.text_encoder import CLIPTextEncoder

logger = logging.getLogger(__name__)

_NO_DECAY_SUFFIXES = ("a_log", "d_skip")


_NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)


############################################################
#       ParameterPartition
############################################################
@dataclass(frozen=True)
class ParameterPartition:
    decay: tuple[nn.Parameter, ...]
    no_decay: tuple[nn.Parameter, ...]

    def optimizer_groups(self, lr: float) -> list[dict]:
        groups: list[dict] = []
        if self.decay:
            groups.append({"params": list(self.decay), "lr": lr})
        if self.no_decay:
            groups.append({"params": list(self.no_decay), "lr": lr, "weight_decay": 0.0})
        return groups

    def assert_complete(self, expected: list[nn.Parameter]) -> None:
        expected_ids = {id(parameter) for parameter in expected if parameter.requires_grad}
        decay_ids = {id(parameter) for parameter in self.decay}
        no_decay_ids = {id(parameter) for parameter in self.no_decay}
        if decay_ids & no_decay_ids:
            raise RuntimeError("optimizer parameter partitions overlap")
        if decay_ids | no_decay_ids != expected_ids:
            raise RuntimeError("optimizer parameter partitions do not cover trainable parameters")


def undecayed_parameter_ids(module: nn.Module) -> set[int]:
    ids: set[int] = set()
    for submodule in module.modules():
        is_norm = isinstance(submodule, (nn.Embedding, *_NORM_TYPES))
        if is_norm or type(submodule).__name__ == "RMSNorm":
            ids.update(id(parameter) for parameter in submodule.parameters(recurse=False))
    return ids


def partition_parameters(
    module: nn.Module, only: list[nn.Parameter] | None = None
) -> ParameterPartition:
    allowed = None if only is None else {id(parameter) for parameter in only}
    exempt = undecayed_parameter_ids(module)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or (allowed is not None and id(parameter) not in allowed):
            continue
        if parameter.ndim <= 1 or id(parameter) in exempt or name.endswith(_NO_DECAY_SUFFIXES):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    partition = ParameterPartition(tuple(decay), tuple(no_decay))
    partition.assert_complete(list(module.parameters()) if only is None else only)
    return partition


def freeze_generator_blocks(generator: MotionTokenGenerator, policy: GeneratorFreeze) -> None:
    if policy == GeneratorFreeze.NONE:
        return
    blocks = list(cast("nn.ModuleList", generator.backbone.blocks))
    if policy == GeneratorFreeze.BACKBONE:
        generator.backbone.requires_grad_(False)
        return
    for block in blocks[: len(blocks) // 2]:
        block.requires_grad_(False)


def warmup_cosine_factor(step: int, warmup: int, total_steps: int, floor: float) -> float:
    warmup = max(warmup, 1)
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return floor + (1.0 - floor) * cosine


def check_schedule_horizon(saved_steps: int | None, total_steps: int | None, extend: bool) -> None:
    if saved_steps == total_steps:
        return
    if not extend:
        raise ValueError(
            f"the resume file has a schedule of {saved_steps} optimizer steps and this run has "
            f"{total_steps}; pass --extend_schedule to change the horizon on purpose"
        )
    logger.warning(
        f"schedule horizon changes from {saved_steps} to {total_steps} optimizer steps; "
        "the learning rate jumps to the value of the new schedule"
    )


def adamw_groups(
    module: nn.Module,
    lr: float,
    use_decay_groups: bool,
    only: list[nn.Parameter] | None = None,
) -> list[dict]:
    parameters = list(module.parameters()) if only is None else list(only)
    if not use_decay_groups:
        return [{"params": parameters, "lr": lr}]
    return partition_parameters(module, only=only).optimizer_groups(lr)


def build_weighter(
    token_generator: MotionTokenGenerator, training_config: GeneratorTrainingConfig
) -> UncertaintyWeighter | None:
    if training_config.loss_weighting != LossWeighting.UNCERTAINTY:
        return None
    device = next(token_generator.parameters()).device
    return UncertaintyWeighter(training_config.loss_weights.active_terms()).to(device)


def build_param_groups(
    token_generator: MotionTokenGenerator,
    text_encoder: CLIPTextEncoder | None,
    training_config: GeneratorTrainingConfig,
    weighter: UncertaintyWeighter | None,
) -> tuple[list[dict], list[nn.Parameter]]:
    groups: list[dict] = []
    clip_params = list(token_generator.parameters())
    groups.extend(adamw_groups(token_generator, training_config.lr, training_config.decay_groups))
    if text_encoder is not None:
        encoder_trainable = [p for p in text_encoder.parameters() if p.requires_grad]
        if encoder_trainable:
            groups.extend(
                adamw_groups(
                    text_encoder,
                    training_config.text_encoder_lr,
                    training_config.decay_groups,
                    only=encoder_trainable,
                )
            )
            clip_params += encoder_trainable
    if weighter is not None:
        groups.append({"params": list(weighter.parameters()), "weight_decay": 0.0})
        clip_params += list(weighter.parameters())
    return groups, clip_params


def build_optimizer(
    token_generator: MotionTokenGenerator,
    text_encoder: CLIPTextEncoder | None,
    training_config: GeneratorTrainingConfig,
    weighter: UncertaintyWeighter | None,
) -> tuple[torch.optim.AdamW, list[nn.Parameter]]:
    freeze_generator_blocks(token_generator, training_config.freeze)
    groups, clip_params = build_param_groups(
        token_generator, text_encoder, training_config, weighter
    )
    optimizer = torch.optim.AdamW(
        groups, lr=training_config.lr, weight_decay=training_config.weight_decay
    )
    return optimizer, clip_params
