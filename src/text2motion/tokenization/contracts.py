from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import torch
from pydantic import Field

from text2motion.motion.contracts import CommandRequest, CorpusLayout, MotionDataConfig
from text2motion.motion.representation import DIM

DROPPED_CODE = -1
_MOTION_FRAMES = MotionDataConfig().max_motion_len


class FiniteScalarStackLayout(StrEnum):
    GROUPED = "grouped"
    RESIDUAL = "residual"


class TokenizerKind(StrEnum):
    FINITE_SCALAR = "fsq"
    RESIDUAL_VECTOR = "rvq"


@dataclass(frozen=True)
class TokenizerConfig:
    kind: TokenizerKind = TokenizerKind.FINITE_SCALAR
    in_dim: int = DIM
    width: int = 512
    downsample: int = 4
    num_quantizers: int = 6
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    quantizer: FiniteScalarStackLayout = FiniteScalarStackLayout.GROUPED
    quant_dropout: float = 0.2
    n_resblocks: int = 3


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


class TokenizerReconstruction(NamedTuple):
    reconstruction: torch.Tensor
    token_indices: torch.Tensor
    commitment_loss: torch.Tensor


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
    accept_legacy_resume: bool = False


@dataclass(frozen=True)
class TokenPackRequest:
    features_dir: Path
    out_path: Path
    segment_frames: int = _MOTION_FRAMES
    stride: int = _MOTION_FRAMES


class TokenizeRequest(CommandRequest):
    features_dir: Path = Field(
        default=CorpusLayout().amass_features_dir,
        description="folder of 263-feature clips to encode",
    )
    tokenizer_ckpt: Path | None = Field(
        default=None,
        description="frozen tokenizer checkpoint; default from the config",
    )
    token_pack: Path = Field(
        default=CorpusLayout().amass_token_pack,
        description="output .npz file of tokens; pretraining reads this file",
    )
    segment_frames: int = Field(
        default=_MOTION_FRAMES,
        description="window length in frames; clips shorter than one window are not encoded",
    )
    stride: int = Field(
        default=_MOTION_FRAMES,
        description="step between windows; a smaller stride gives more, overlapping windows",
    )
    regen_dir: Path = Field(
        default=CorpusLayout().regen_dir,
        description="folder with pose_data; the heldout clips are found and excluded through it",
    )


class TrainTokenizerRequest(CommandRequest):
    tokenizer: TokenizerKind = Field(
        description="quantizer family: fsq has no learned codebook, rvq is the baseline"
    )
    epochs: int = Field(
        default=TokenizerTrainingRequest().epochs, description="passes over the training set"
    )
    window: int = Field(
        default=TokenizerTrainingRequest().window,
        description="training crop in frames; a longer crop gives more context and uses more memory",
    )
    batch_size: int = Field(
        default=TokenizerTrainingRequest().batch_size,
        description="clips per step; limited by GPU memory",
    )
    ema_decay: float = Field(
        default=TokenizerTrainingRequest().ema_decay,
        description="decay of the averaged weights that are evaluated; higher is smoother and slower",
    )
    eval_every: int = Field(
        default=TokenizerTrainingRequest().eval_every,
        description="epochs between reconstruction evaluations",
    )
    max_eval_clips: int | None = Field(
        default=TokenizerTrainingRequest().max_eval_clips,
        description="cap on evaluation clips; a low cap is faster and gives a noisier FID",
    )
    num_workers: int = Field(
        default=TokenizerTrainingRequest().num_workers,
        description="data loader processes; 0 loads in the training process and needs the least RAM",
    )
    ckpt_name: str | None = Field(
        default=None,
        description="checkpoint file name inside the checkpoint folder; also names the run",
    )
    resume: bool = Field(
        default=False, description="continue from the saved state of this checkpoint"
    )
    accept_legacy_resume: bool = Field(
        default=False,
        description="with --resume: load an old resume file; its RNG state starts fresh",
    )
