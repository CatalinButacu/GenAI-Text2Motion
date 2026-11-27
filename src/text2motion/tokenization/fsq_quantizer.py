from __future__ import annotations

import math

import torch
from torch import nn

from text2motion.tokenization.quantizer_base import (
    Quantizer,
    QuantizerStack,
    ResidualQuantizerStack,
    check_stage_count,
    dropped_code_indices,
    typed_modules,
)


def reject_other_fsq_levels(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    built = module.get_buffer("levels")
    stored = state_dict.get(f"{prefix}levels")
    if stored is None or torch.equal(stored.to(built), built):
        return
    error_msgs.append(
        f"{prefix}levels: the checkpoint has {stored.tolist()}, but the config builds "
        f"{built.tolist()}; the FSQ bounds come from the config, so the tokens would change; "
        "load the checkpoint with the config it was trained with"
    )


def check_fsq_levels(levels: tuple[int, ...]) -> None:
    if not levels or min(levels) < 3:
        raise ValueError(
            f"FSQ levels must be a non-empty list of values of at least 3, got {levels}; "
            "a level of 1 or 2 gives NaN codes with this boundary formula"
        )


def register_level_buffers(
    module: nn.Module, levels: tuple[int, ...], level_values: torch.Tensor
) -> None:
    module.register_buffer("levels", level_values)
    module.register_buffer("half_l_bound", (level_values - 1) * 0.5)
    module.register_buffer("half_width", torch.div(level_values, 2, rounding_mode="floor"))
    basis = torch.cumprod(torch.tensor([1] + list(levels[:-1]), dtype=torch.long), dim=0)
    module.register_buffer("basis", basis)


def register_bound_buffers(
    module: nn.Module, level_values: torch.Tensor, boundary_margin: float
) -> None:
    bound_scale = (level_values - 1) * 0.5 * (1 - boundary_margin)
    bound_offset = torch.where(level_values % 2 == 0, 0.5, 0.0)
    bound_shift = torch.atanh(bound_offset / bound_scale)
    module.register_buffer("bound_scale", bound_scale, persistent=False)
    module.register_buffer("bound_offset", bound_offset, persistent=False)
    module.register_buffer("bound_shift", bound_shift, persistent=False)


def check_code_count(levels: tuple[int, ...], codebook_size: int) -> None:
    if codebook_size > 2**62:
        raise ValueError(f"FSQ levels {levels} give {codebook_size} codes; too many for int64")


def check_reachable_digits(
    module: nn.Module,
    levels: tuple[int, ...],
    level_values: torch.Tensor,
    boundary_margin: float,
) -> None:
    bound_scale = module.get_buffer("bound_scale")
    bound_offset = module.get_buffer("bound_offset")
    reachable = torch.round(bound_scale - bound_offset) - torch.round(-bound_scale - bound_offset)
    if bool((reachable < level_values - 1).any()):
        raise ValueError(
            f"FSQ levels {levels}: the boundary margin {boundary_margin} makes an end digit "
            "unreachable; use smaller levels"
        )


############################################################
#       FiniteScalarQuantizer
############################################################
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
        check_fsq_levels(levels)
        level_values = torch.tensor(levels, dtype=torch.float32)
        register_level_buffers(self, levels, level_values)
        register_bound_buffers(self, level_values, boundary_margin)

        self.dimension_count = len(levels)
        self.codebook_size = math.prod(levels)
        check_code_count(levels, self.codebook_size)
        check_reachable_digits(self, levels, level_values, boundary_margin)
        self.register_load_state_dict_pre_hook(reject_other_fsq_levels)

    def bound(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent + self.bound_shift) * self.bound_scale - self.bound_offset

    def quantize(self, latent: torch.Tensor) -> torch.Tensor:
        def round_straight_through(latent: torch.Tensor) -> torch.Tensor:
            return latent + (torch.round(latent) - latent).detach()

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


############################################################
#       ResidualFiniteScalarQuantizer
############################################################
class ResidualFiniteScalarQuantizer(ResidualQuantizerStack):
    def __init__(
        self, levels: tuple[int, ...], num_quantizers: int, dropout_probability: float
    ) -> None:
        super().__init__()
        stage_count = check_stage_count(num_quantizers)
        self.layers = nn.ModuleList([FiniteScalarQuantizer(levels) for _ in range(stage_count)])
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


############################################################
#       GroupedFiniteScalarQuantizer
############################################################
class GroupedFiniteScalarQuantizer(QuantizerStack):
    def __init__(self, levels: tuple[int, ...], num_quantizers: int) -> None:
        super().__init__()
        group_count = check_stage_count(num_quantizers)
        self.groups = nn.ModuleList([FiniteScalarQuantizer(levels) for _ in range(group_count)])

    @property
    def quantizers(self) -> list[Quantizer]:
        return typed_modules(self.groups, Quantizer)

    def combine_code_vectors(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(codes, dim=-1)

    def shared_quantizer(self) -> FiniteScalarQuantizer:
        return typed_modules(self.groups, FiniteScalarQuantizer)[0]

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantizer = self.shared_quantizer()
        grouped = latent.unflatten(-1, (len(self.groups), quantizer.dimension_count))
        codes = quantizer.quantize(grouped)
        indices = quantizer.codes_to_indices(codes)
        return codes.flatten(-2), indices, latent.new_zeros(())

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        checked = self.checked_indices(indices)
        return self.shared_quantizer().indices_to_codes(checked).flatten(-2)
