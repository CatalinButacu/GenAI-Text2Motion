from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.tokenization.contracts import ResidualVectorQuantizationConfig
from text2motion.tokenization.quantizer_base import (
    Quantizer,
    ResidualQuantizerStack,
    check_stage_count,
    dropped_code_indices,
    typed_modules,
)


############################################################
#       VectorQuantizer
############################################################
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
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def distances(self, flat_latents: torch.Tensor) -> torch.Tensor:
        return (
            flat_latents.pow(2).sum(1, keepdim=True)
            - 2 * flat_latents @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )

    def ema_update(self, flat_latents: torch.Tensor, assignments: torch.Tensor) -> None:
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
        self.reset_dead_codes(flat_latents)

    def reset_dead_codes(self, flat_latents: torch.Tensor) -> None:
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
            indices = self.distances(flat_latents).argmin(1)
            assignments = F.one_hot(indices, self.codebook_size).type_as(flat_latents)

            quantized = (assignments @ self.embed).view_as(latent)
            if self.training:
                self.ema_update(flat_latents, assignments)

        commitment_loss = F.mse_loss(latent, quantized.detach())
        return quantized, indices.view(latent.shape[:-1]), commitment_loss

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)


############################################################
#       ResidualVectorQuantizer
############################################################
class ResidualVectorQuantizer(ResidualQuantizerStack):
    def __init__(self, config: ResidualVectorQuantizationConfig) -> None:
        super().__init__()
        self.dropout_probability = config.quant_dropout
        self.layers = nn.ModuleList(
            VectorQuantizer(
                config.codebook_size, config.code_dim, config.ema_decay, config.reset_threshold
            )
            for _ in range(check_stage_count(config.num_quantizers))
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
