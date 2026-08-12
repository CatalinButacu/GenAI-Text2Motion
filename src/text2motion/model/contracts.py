from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import nn

DROPPED_CODE = -1


class TokenizerOutput(NamedTuple):
    recon: torch.Tensor
    indices: torch.Tensor
    commit: torch.Tensor


class MotionQuantizer(nn.Module, ABC):
    @property
    @abstractmethod
    def units(self) -> nn.ModuleList: ...

    @abstractmethod
    def combine_codes(self, codes: list[torch.Tensor]) -> torch.Tensor: ...

    @abstractmethod
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor: ...


class MotionTokenizer(nn.Module, ABC):
    @property
    @abstractmethod
    def codebook_size(self) -> int: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, indices: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> TokenizerOutput: ...


def reject_dropped_codes(indices: torch.Tensor) -> torch.Tensor:
    if bool((indices < 0).any()):
        raise RuntimeError(
            "quantizer dropout left inactive levels in this batch, so the emitted indices do not "
            "describe the motion. Call encode() in eval mode, or disable quant_dropout, before "
            "tokenizing a corpus or training a generator on these tokens."
        )
    return indices
