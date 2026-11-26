from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

DT_MIN = 1e-3
DT_MAX = 1e-1
DT_FLOOR = 1e-4


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


def init_reference_dt(dt_proj: nn.Linear, dt_rank: int) -> None:
    bound = dt_rank**-0.5
    span = math.log(DT_MAX) - math.log(DT_MIN)
    with torch.no_grad():
        nn.init.uniform_(dt_proj.weight, -bound, bound)
        dt = torch.exp(torch.rand(dt_proj.out_features) * span + math.log(DT_MIN))
        dt = dt.clamp(min=DT_FLOOR)
        dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))


def parallel_scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    length = a.size(1)
    shift = 1
    while shift < length:
        pad = (0, 0, 0, 0, shift, 0)
        a_prev = F.pad(a[:, :-shift], pad, value=1.0)
        x_prev = F.pad(x[:, :-shift], pad, value=0.0)
        x = x + a * x_prev
        a = a * a_prev
        shift *= 2
    return x
