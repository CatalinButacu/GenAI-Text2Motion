from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.tokenization.contracts import (
    DROPPED_CODE,
    ResidualVectorQuantizationConfig,
)


def typed_modules[ModuleType: nn.Module](
    modules: nn.ModuleList, kind: type[ModuleType]
) -> list[ModuleType]:
    return [module for module in modules if isinstance(module, kind)]


class Quantizer(nn.Module, ABC):
    codebook_size: int

    @abstractmethod
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor: ...


class QuantizerStack(nn.Module, ABC):
    @property
    @abstractmethod
    def quantizers(self) -> list[Quantizer]: ...

    @abstractmethod
    def combine_code_vectors(self, codes: list[torch.Tensor]) -> torch.Tensor: ...

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


def dropped_code_indices(latent: torch.Tensor) -> torch.Tensor:
    return torch.full(latent.shape[:-1], DROPPED_CODE, dtype=torch.long, device=latent.device)


def round_straight_through(latent: torch.Tensor) -> torch.Tensor:
    return latent + (torch.round(latent) - latent).detach()


class FiniteScalarQuantizer(Quantizer):
    levels: torch.Tensor
    half_l_bound: torch.Tensor
    half_width: torch.Tensor
    basis: torch.Tensor
    bound_scale: torch.Tensor
    bound_offset: torch.Tensor
    bound_shift: torch.Tensor

    def __init__(self, levels: tuple[int, ...], boundary_margin: float = 1e-3) -> None:
        super().__init__()
        level_values = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", level_values)
        self.register_buffer("half_l_bound", (level_values - 1) * 0.5)
        self.register_buffer("half_width", torch.div(level_values, 2, rounding_mode="floor"))
        basis = torch.cumprod(torch.tensor([1] + list(levels[:-1]), dtype=torch.long), dim=0)
        self.register_buffer("basis", basis)

        bound_scale = (level_values - 1) * 0.5 * (1 - boundary_margin)
        bound_offset = torch.where(level_values % 2 == 0, 0.5, 0.0)
        bound_shift = torch.atanh(bound_offset / bound_scale)
        self.register_buffer("bound_scale", bound_scale, persistent=False)
        self.register_buffer("bound_offset", bound_offset, persistent=False)
        self.register_buffer("bound_shift", bound_shift, persistent=False)

        self.dimension_count = len(levels)
        self.codebook_size = int(torch.prod(level_values).item())

    def bound(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent + self.bound_shift) * self.bound_scale - self.bound_offset

    def quantize(self, latent: torch.Tensor) -> torch.Tensor:
        quantized = round_straight_through(self.bound(latent))
        return quantized / self.half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        shifted = (codes * self.half_width) + self.half_width
        return (shifted.round().long() * self.basis).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        index_column = indices.unsqueeze(-1)
        levels = self.levels.long()
        digits = (index_column // self.basis) % levels
        return (digits.float() - self.half_width) / self.half_width


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
        quantizers = self.quantizers
        code_vectors = quantizers[0].indices_to_codes(indices[..., 0])
        for stage, quantizer in enumerate(quantizers[1:], start=1):
            code_vectors = code_vectors + quantizer.indices_to_codes(indices[..., stage])
        return code_vectors


class ResidualFiniteScalarQuantizer(ResidualQuantizerStack):
    def __init__(
        self, levels: tuple[int, ...], num_quantizers: int, dropout_probability: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FiniteScalarQuantizer(levels) for _ in range(num_quantizers)])
        self.dropout_probability = dropout_probability

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = latent
        quantized = torch.zeros_like(latent)
        indices: list[torch.Tensor] = []
        active_stages = self.active_stage_count()

        for stage, layer in enumerate(typed_modules(self.layers, FiniteScalarQuantizer)):
            if stage < active_stages:
                code = layer.quantize(residual)
                residual = residual - code
                quantized = quantized + code
                indices.append(layer.codes_to_indices(code))
            else:
                indices.append(dropped_code_indices(latent))

        return quantized, torch.stack(indices, dim=-1), latent.new_zeros(())


class GroupedFiniteScalarQuantizer(QuantizerStack):
    def __init__(self, levels: tuple[int, ...], num_quantizers: int) -> None:
        super().__init__()
        self.groups = nn.ModuleList([FiniteScalarQuantizer(levels) for _ in range(num_quantizers)])

    @property
    def quantizers(self) -> list[Quantizer]:
        return typed_modules(self.groups, Quantizer)

    def combine_code_vectors(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(codes, dim=-1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codes: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        groups = typed_modules(self.groups, FiniteScalarQuantizer)
        chunks = latent.split([group.dimension_count for group in groups], dim=-1)
        for group, chunk in zip(groups, chunks, strict=True):
            code = group.quantize(chunk)
            codes.append(code)
            indices.append(group.codes_to_indices(code))
        return torch.cat(codes, dim=-1), torch.stack(indices, dim=-1), latent.new_zeros(())

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        groups = typed_modules(self.groups, FiniteScalarQuantizer)
        return torch.cat(
            [group.indices_to_codes(indices[..., index]) for index, group in enumerate(groups)],
            dim=-1,
        )


class VectorQuantizer(Quantizer):
    embed: torch.Tensor
    cluster_size: torch.Tensor
    embed_avg: torch.Tensor

    def __init__(
        self, codebook_size: int, code_width: int, ema_decay: float, reset_threshold: float
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.code_width = code_width
        self.decay = ema_decay
        self.reset_threshold = reset_threshold
        self.count_smoothing = 1e-5

        embed = torch.randn(codebook_size, code_width)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def __distances(self, flat_latents: torch.Tensor) -> torch.Tensor:
        return (
            flat_latents.pow(2).sum(1, keepdim=True)
            - 2 * flat_latents @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )

    def __ema_update(self, flat_latents: torch.Tensor, assignments: torch.Tensor) -> None:
        assignment_counts = assignments.sum(0)
        assigned_sum = assignments.t() @ flat_latents
        self.cluster_size.mul_(self.decay).add_(assignment_counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(assigned_sum, alpha=1 - self.decay)

        total = self.cluster_size.sum()
        smoothing = self.count_smoothing
        smoothed = (
            (self.cluster_size + smoothing) / (total + self.codebook_size * smoothing) * total
        )
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
        self.__reset_dead_codes(flat_latents)

    def __reset_dead_codes(self, flat_latents: torch.Tensor) -> None:
        dead = self.cluster_size < self.reset_threshold
        dead_count = int(dead.sum().item())
        if dead_count == 0:
            return

        rows = torch.randint(0, flat_latents.size(0), (dead_count,), device=flat_latents.device)
        self.embed[dead] = flat_latents[rows]
        self.embed_avg[dead] = flat_latents[rows]
        self.cluster_size[dead] = 1.0

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            flat_latents = latent.reshape(-1, self.code_width)
            indices = self.__distances(flat_latents).argmin(1)
            assignments = F.one_hot(indices, self.codebook_size).type_as(flat_latents)

            if self.training:
                self.__ema_update(flat_latents, assignments)

        quantized = (assignments @ self.embed).view_as(latent)
        commitment_loss = F.mse_loss(latent, quantized.detach())
        return quantized, indices.view(latent.shape[:-1]), commitment_loss

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)


class ResidualVectorQuantizer(ResidualQuantizerStack):
    def __init__(self, config: ResidualVectorQuantizationConfig) -> None:
        super().__init__()
        self.dropout_probability = config.quant_dropout
        self.layers = nn.ModuleList(
            VectorQuantizer(
                config.codebook_size, config.code_dim, config.ema_decay, config.reset_threshold
            )
            for _ in range(config.num_quantizers)
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = latent
        quantized = torch.zeros_like(latent)
        commitment_loss = latent.new_zeros(())
        indices: list[torch.Tensor] = []
        active_stages = self.active_stage_count()

        for stage, layer in enumerate(typed_modules(self.layers, VectorQuantizer)):
            if stage < active_stages:
                code, stage_indices, stage_commitment = layer(residual)
                residual = residual - code
                quantized = quantized + code
                commitment_loss = commitment_loss + stage_commitment
                indices.append(stage_indices)
            else:
                indices.append(dropped_code_indices(latent))

        if self.training:
            quantized = latent + (quantized - latent).detach()

        return quantized, torch.stack(indices, dim=-1), commitment_loss
