from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import torch

from text2motion.motion.representation import DIM

DROPPED_CODE = -1


############################################################
#       FiniteScalarStackLayout
############################################################
class FiniteScalarStackLayout(StrEnum):
    GROUPED = "grouped"
    RESIDUAL = "residual"


############################################################
#       TokenizerKind
############################################################
class TokenizerKind(StrEnum):
    FINITE_SCALAR = "fsq"
    RESIDUAL_VECTOR = "rvq"


def check_network_ranges(*sizes: int) -> None:
    if any(size < 1 for size in sizes):
        raise ValueError(f"network sizes must be positive, got {sizes}")


def check_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def check_not_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")


############################################################
#       TokenizerConfig
############################################################
@dataclass(frozen=True)
class TokenizerConfig:
    in_dim: int = DIM
    width: int = 512
    downsample: int = 4
    num_quantizers: int = 6
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    kind: TokenizerKind = TokenizerKind.FINITE_SCALAR
    quantizer: FiniteScalarStackLayout = FiniteScalarStackLayout.GROUPED
    quant_dropout: float = 0.2
    n_resblocks: int = 3
    causal: bool = False
    heading_correction: bool = False
    heading_weight: float = 0.1

    def __post_init__(self) -> None:
        check_network_ranges(self.in_dim, self.width, self.downsample, self.num_quantizers)
        check_fraction("quant_dropout", self.quant_dropout)
        check_not_negative("n_resblocks", self.n_resblocks)
        check_not_negative("heading_weight", self.heading_weight)


############################################################
#       ResidualVectorQuantizationConfig
############################################################
@dataclass(frozen=True)
class ResidualVectorQuantizationConfig:
    in_dim: int = DIM
    width: int = 512
    downsample: int = 4
    n_resblocks: int = 3
    num_quantizers: int = 6
    codebook_size: int = 512
    code_dim: int = 512
    ema_decay: float = 0.99
    commitment_beta: float = 0.02
    quant_dropout: float = 0.2
    reset_threshold: float = 1.0

    def __post_init__(self) -> None:
        check_network_ranges(self.in_dim, self.width, self.downsample, self.num_quantizers)
        check_network_ranges(self.codebook_size, self.code_dim, 1, 1)
        check_fraction("quant_dropout", self.quant_dropout)
        check_fraction("ema_decay", self.ema_decay)
        check_not_negative("n_resblocks", self.n_resblocks)
        check_not_negative("commitment_beta", self.commitment_beta)
        check_not_negative("reset_threshold", self.reset_threshold)


############################################################
#       TokenizerReconstruction
############################################################
class TokenizerReconstruction(NamedTuple):
    reconstruction: torch.Tensor
    token_indices: torch.Tensor
    commitment_loss: torch.Tensor


############################################################
#       TokenPackPlan
############################################################
@dataclass(frozen=True)
class TokenPackPlan:
    features_dir: Path
    token_pack: Path
    segment_frames: int
    stride: int


############################################################
#       TokenizerSchedule
############################################################
@dataclass(frozen=True)
class TokenizerSchedule:
    epochs: int
    eval_every: int
    resume: bool = False
    accept_legacy_resume: bool = False
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.max_steps is not None and self.resume:
            raise ValueError("max_steps counts from the first step; it cannot resume a run")
