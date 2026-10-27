from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.tokenization.contracts import (
    DROPPED_CODE,
    RvqConfig,
)

_Module = TypeVar("_Module", bound=nn.Module)


def typed_modules(modules: nn.ModuleList, kind: type[_Module]) -> list[_Module]:
    return [module for module in modules if isinstance(module, kind)]


class QuantizerUnit(nn.Module, ABC):
    codebook_size: int

    @abstractmethod
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor: ...


class LatentQuantizer(nn.Module, ABC):
    @property
    @abstractmethod
    def quantizer_units(self) -> list[QuantizerUnit]: ...

    @abstractmethod
    def compose_latent_codes(self, codes: list[torch.Tensor]) -> torch.Tensor: ...

    @abstractmethod
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor: ...


def reject_dropped_codes(indices: torch.Tensor) -> torch.Tensor:
    if bool((indices < 0).any()):
        raise RuntimeError(
            "quantizer dropout left inactive levels in this batch, so the emitted indices do not "
            "describe the motion. Call encode() in eval mode, or disable quant_dropout, before "
            "tokenizing a corpus or training a generator on these tokens."
        )
    return indices


def dropped_code_indices(z: torch.Tensor) -> torch.Tensor:
    return torch.full(z.shape[:-1], DROPPED_CODE, dtype=torch.long, device=z.device)


def round_ste(z: torch.Tensor) -> torch.Tensor:
    return z + (torch.round(z) - z).detach()


class FSQ(QuantizerUnit):
    levels: torch.Tensor
    half_l_bound: torch.Tensor
    half_width: torch.Tensor
    basis: torch.Tensor
    bound_scale: torch.Tensor
    bound_offset: torch.Tensor
    bound_shift: torch.Tensor

    def __init__(self, levels: tuple[int, ...], eps: float = 1e-3) -> None:
        super().__init__()
        levels_t = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", levels_t)
        self.register_buffer("half_l_bound", (levels_t - 1) * 0.5)
        self.register_buffer("half_width", torch.div(levels_t, 2, rounding_mode="floor"))
        basis = torch.cumprod(torch.tensor([1] + list(levels[:-1]), dtype=torch.long), dim=0)
        self.register_buffer("basis", basis)

        half_l = (levels_t - 1) * 0.5 * (1 - eps)
        offset = torch.where(levels_t % 2 == 0, 0.5, 0.0)
        self.register_buffer("bound_scale", half_l, persistent=False)
        self.register_buffer("bound_offset", offset, persistent=False)
        self.register_buffer("bound_shift", torch.atanh(offset / half_l), persistent=False)

        self.dim = len(levels)
        self.codebook_size = int(torch.prod(levels_t).item())

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z + self.bound_shift) * self.bound_scale - self.bound_offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        quantized = round_ste(self.bound(z))
        return quantized / self.half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        shifted = (codes * self.half_width) + self.half_width
        return (shifted.round().long() * self.basis).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.unsqueeze(-1)
        levels = self.levels.long()
        digits = (idx // self.basis) % levels
        return (digits.float() - self.half_width) / self.half_width


class ResidualQuantizerStack(LatentQuantizer, ABC):
    layers: nn.ModuleList
    num_quantizers: int
    dropout_p: float

    @property
    def quantizer_units(self) -> list[QuantizerUnit]:
        return typed_modules(self.layers, QuantizerUnit)

    def compose_latent_codes(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(codes, dim=0).sum(0)

    def active_levels(self) -> int:
        if self.training and self.dropout_p > 0 and torch.rand(()) < self.dropout_p:
            return int(torch.randint(1, self.num_quantizers + 1, ()).item())

        return self.num_quantizers

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        units = self.quantizer_units
        out = units[0].indices_to_codes(indices[..., 0])
        for stage, unit in enumerate(units[1:], start=1):
            out = out + unit.indices_to_codes(indices[..., stage])
        return out


class ResidualFSQ(ResidualQuantizerStack):
    def __init__(self, levels: tuple[int, ...], num_quantizers: int, dropout_p: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FSQ(levels) for _ in range(num_quantizers)])
        self.num_quantizers = num_quantizers
        self.dropout_p = dropout_p

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        indices: list[torch.Tensor] = []
        n_active = self.active_levels()

        for i, layer in enumerate(typed_modules(self.layers, FSQ)):
            if i < n_active:
                code = layer.quantize(residual)
                residual = residual - code
                quantized = quantized + code
                indices.append(layer.codes_to_indices(code))
            else:
                indices.append(dropped_code_indices(z))

        return quantized, torch.stack(indices, dim=-1)


class GroupedFSQ(LatentQuantizer):
    def __init__(self, levels: tuple[int, ...], num_quantizers: int) -> None:
        super().__init__()
        self.groups = nn.ModuleList([FSQ(levels) for _ in range(num_quantizers)])
        self.num_quantizers = num_quantizers
        self.dim = len(levels)

    @property
    def quantizer_units(self) -> list[QuantizerUnit]:
        return typed_modules(self.groups, QuantizerUnit)

    def compose_latent_codes(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(codes, dim=-1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for group_index, group in enumerate(typed_modules(self.groups, FSQ)):
            chunk = z[..., group_index * self.dim : (group_index + 1) * self.dim]
            code = group.quantize(chunk)
            codes.append(code)
            indices.append(group.codes_to_indices(code))
        return torch.cat(codes, dim=-1), torch.stack(indices, dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                group.indices_to_codes(indices[..., i])
                for i, group in enumerate(typed_modules(self.groups, FSQ))
            ],
            dim=-1,
        )


class EmaVectorQuantizer(QuantizerUnit):
    embed: torch.Tensor
    cluster_size: torch.Tensor
    embed_avg: torch.Tensor

    def __init__(
        self, codebook_size: int, code_dim: int, ema_decay: float, reset_threshold: float
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.decay = ema_decay
        self.reset_threshold = reset_threshold
        self.eps = 1e-5

        embed = torch.randn(codebook_size, code_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def __distances(self, flat: torch.Tensor) -> torch.Tensor:
        return (
            flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1)
        )

    def __ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        n = onehot.sum(0)
        embed_sum = onehot.t() @ flat
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + self.codebook_size * self.eps) * total
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
        self.__reset_dead_codes(flat)

    def __reset_dead_codes(self, flat: torch.Tensor) -> None:
        dead = self.cluster_size < self.reset_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return

        pick = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
        self.embed[dead] = flat[pick]
        self.embed_avg[dead] = flat[pick]
        self.cluster_size[dead] = 1.0

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            flat = z.reshape(-1, self.code_dim)
            indices = self.__distances(flat).argmin(1)
            onehot = F.one_hot(indices, self.codebook_size).type_as(flat)

            if self.training:
                self.__ema_update(flat, onehot)

        quant = (onehot @ self.embed).view_as(z)
        commit = F.mse_loss(z, quant.detach())
        return quant, indices.view(z.shape[:-1]), commit

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self.lookup(indices)


class ResidualVQ(ResidualQuantizerStack):
    def __init__(self, cfg: RvqConfig) -> None:
        super().__init__()
        self.num_quantizers = cfg.num_quantizers
        self.dropout_p = cfg.quant_dropout
        self.layers = nn.ModuleList(
            EmaVectorQuantizer(cfg.codebook_size, cfg.code_dim, cfg.ema_decay, cfg.reset_threshold)
            for _ in range(cfg.num_quantizers)
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        commit = z.new_zeros(())
        indices: list[torch.Tensor] = []
        n_active = self.active_levels()

        for i, layer in enumerate(typed_modules(self.layers, EmaVectorQuantizer)):
            if i < n_active:
                code, idx, c = layer(residual)
                residual = residual - code
                quantized = quantized + code
                commit = commit + c
                indices.append(idx)
            else:
                indices.append(dropped_code_indices(z))

        if self.training:
            quantized = z + (quantized - z).detach()

        return quantized, torch.stack(indices, dim=-1), commit
