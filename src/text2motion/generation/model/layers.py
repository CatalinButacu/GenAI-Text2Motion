from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


############################################################
#       RMSNorm
############################################################
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


EMBEDDING_STD = 0.02


def init_reference_embedding(embedding: nn.Embedding) -> None:
    nn.init.normal_(embedding.weight, std=EMBEDDING_STD)


def zero_linear_bias(layer: nn.Linear) -> None:
    if layer.bias is None:
        raise ValueError("the reference init zeros a bias that this layer does not have")
    nn.init.zeros_(layer.bias)


def init_residual_output(layer: nn.Linear, residual_count: int) -> None:
    if residual_count < 1:
        raise ValueError(f"residual_count must be >= 1, got {residual_count}")
    with torch.no_grad():
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        layer.weight.div_(math.sqrt(residual_count))


############################################################
#       SequenceBackbone
############################################################
class SequenceBackbone(nn.Module, ABC):
    @property
    @abstractmethod
    def max_positions(self) -> int | None: ...

    @abstractmethod
    def init_state(self, batch: int, device: torch.device) -> list: ...

    @abstractmethod
    def step(self, x_t: torch.Tensor, state: list) -> tuple[torch.Tensor, list]: ...
