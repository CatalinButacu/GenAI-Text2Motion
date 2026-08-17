from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

SMPLX_MODEL_TYPE = "smplx"


class Gender(StrEnum):
    NEUTRAL = "neutral"
    MALE = "male"
    FEMALE = "female"


@dataclass(frozen=True)
class MotionClip:
    features: torch.Tensor
    frame_count: int
    caption: str | None = None
    clip_id: str | None = None


@dataclass(frozen=True)
class MotionBatch:
    features: torch.Tensor
    lengths: torch.Tensor
    captions: list[str]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: str | torch.device) -> MotionBatch:
        return MotionBatch(
            features=self.features.to(device),
            lengths=self.lengths.to(device),
            captions=self.captions,
        )


@dataclass(frozen=True)
class MotionTokens:
    indices: torch.Tensor
    token_count: int
    frame_count: int


@dataclass(frozen=True)
class MotionChunk:
    motion: MotionClip
    index: int
    is_final: bool


@dataclass(frozen=True)
class GeneratedMotion:
    prompt: str
    tokens: MotionTokens
    motion: MotionClip
