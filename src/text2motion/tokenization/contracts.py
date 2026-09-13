from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import torch

from text2motion.motion.contracts import MotionDataConfig
from text2motion.motion.representation import DIM

DROPPED_CODE = -1
_MOTION_FRAMES = MotionDataConfig().max_motion_len


class FsqComposition(StrEnum):
    GROUPED = "grouped"
    RESIDUAL = "residual"


class TokenizerKind(StrEnum):
    FSQ = "fsq"
    RVQ = "rvq"


@dataclass(frozen=True)
class TokenizerConfig:
    kind: TokenizerKind = TokenizerKind.FSQ
    in_dim: int = DIM
    width: int = 512
    downsample: int = 4
    num_quantizers: int = 6
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    quantizer: FsqComposition = FsqComposition.GROUPED
    quant_dropout: float = 0.2
    n_resblocks: int = 3


@dataclass(frozen=True)
class RvqConfig:
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


class TokenizerForwardOutput(NamedTuple):
    recon: torch.Tensor
    indices: torch.Tensor
    commit: torch.Tensor


@dataclass(frozen=True)
class TokenizerTrainingRequest:
    epochs: int = 50
    window: int = 64
    batch_size: int = 128
    ema_decay: float = 0.99
    eval_every: int = 5
    max_eval_clips: int | None = None
    num_workers: int = 0
    checkpoint_name: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class TokenPackRequest:
    features_dir: Path
    out_path: Path
    segment_frames: int = _MOTION_FRAMES
    stride: int = _MOTION_FRAMES
