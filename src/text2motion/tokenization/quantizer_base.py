from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from text2motion.tokenization.contracts import DROPPED_CODE


def typed_modules[ModuleType: nn.Module](
    modules: nn.ModuleList, kind: type[ModuleType]
) -> list[ModuleType]:
    return [module for module in modules if isinstance(module, kind)]


############################################################
#       Quantizer
############################################################
class Quantizer(nn.Module, ABC):
    codebook_size: int

    @abstractmethod
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor: ...


############################################################
#       QuantizerStack
############################################################
class QuantizerStack(nn.Module, ABC):
    @property
    @abstractmethod
    def quantizers(self) -> list[Quantizer]: ...

    @abstractmethod
    def combine_code_vectors(self, codes: list[torch.Tensor]) -> torch.Tensor: ...

    @abstractmethod
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor: ...

    def checked_indices(self, indices: torch.Tensor) -> torch.Tensor:
        quantizers = self.quantizers
        expected = len(quantizers)
        if indices.shape[-1] != expected:
            raise ValueError(
                f"the indices have {indices.shape[-1]} codebooks, but this quantizer stack has "
                f"{expected}; the tokens come from another tokenizer configuration"
            )
        if indices.is_floating_point() or indices.is_complex() or indices.dtype == torch.bool:
            raise TypeError(f"token indices must be an integer tensor, got {indices.dtype}")
        codebook_size = quantizers[0].codebook_size
        if bool(((indices < 0) | (indices >= codebook_size)).any()):
            raise ValueError(
                f"the indices hold values outside 0..{codebook_size - 1}; a dropped code "
                f"({DROPPED_CODE}) or a token of another tokenizer cannot be decoded"
            )
        return indices


def check_stage_count(stage_count: int) -> int:
    if stage_count < 1:
        raise ValueError(f"a quantizer stack needs at least 1 quantizer, got {stage_count}")
    return stage_count


def reject_dropped_codes(indices: torch.Tensor) -> torch.Tensor:
    if bool((indices < 0).any()):
        raise RuntimeError(
            "quantizer dropout left inactive levels in this batch, so the emitted indices do not "
            "describe the motion. Call encode() in eval mode, or disable quant_dropout, before "
            "tokenizing a corpus or training a generator on these tokens."
        )
    return indices


def dropped_code_indices(latent: torch.Tensor) -> torch.Tensor:
    return torch.full(latent.shape[:-1], DROPPED_CODE, dtype=torch.long, device=latent.device)


############################################################
#       ResidualQuantizerStack
############################################################
class ResidualQuantizerStack(QuantizerStack, ABC):
    layers: nn.ModuleList
    dropout_probability: float

    @property
    def quantizers(self) -> list[Quantizer]:
        return typed_modules(self.layers, Quantizer)

    def combine_code_vectors(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(codes, dim=0).sum(0)

    def active_stage_count(self) -> int:
        stage_count = len(self.layers)
        dropout_active = self.training and self.dropout_probability > 0
        if dropout_active and torch.rand(()) < self.dropout_probability:
            return int(torch.randint(1, stage_count + 1, ()).item())

        return stage_count

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        indices = self.checked_indices(indices)
        quantizers = self.quantizers
        code_vectors = quantizers[0].indices_to_codes(indices[..., 0])
        for stage, quantizer in enumerate(quantizers[1:], start=1):
            code_vectors = code_vectors + quantizer.indices_to_codes(indices[..., stage])
        return code_vectors
