import torch
from torch import nn
from torch.nn import functional as F

from text2motion.model.contracts import (
    DROPPED_CODE,
    MotionQuantizer,
    MotionTokenizer,
    TokenizerOutput,
    reject_dropped_codes,
)
from text2motion.model.tokenizer import Decoder1d, Encoder1d
from text2motion.shared.config import RvqBaselineCfg


class EmaVectorQuantizer(nn.Module):
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
        self.register_buffer("embed", embed)  # the codebook (K, D), EMA-updated (not a Parameter)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))  # EMA usage counts
        self.register_buffer("embed_avg", embed.clone())  # EMA of summed assigned vectors

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        return (
            flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1)
        )

    def _ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        n = onehot.sum(0)  # (K,) assignments this batch
        embed_sum = onehot.t() @ flat  # (K, D)
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + self.codebook_size * self.eps) * total
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
        self._reset_dead_codes(flat)

    def _reset_dead_codes(self, flat: torch.Tensor) -> None:
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
            indices = self._distances(flat).argmin(1)  # (N,)
            onehot = F.one_hot(indices, self.codebook_size).type_as(flat)

            if self.training:
                self._ema_update(flat, onehot)

        quant = (onehot @ self.embed).view_as(z)
        commit = F.mse_loss(z, quant.detach())
        return quant, indices.view(z.shape[:-1]), commit

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self.lookup(indices)


class ResidualVQ(MotionQuantizer):
    def __init__(self, cfg: RvqBaselineCfg) -> None:
        super().__init__()
        self.num_quantizers = cfg.num_quantizers
        self.dropout_p = cfg.quant_dropout
        self.layers = nn.ModuleList(
            EmaVectorQuantizer(cfg.codebook_size, cfg.code_dim, cfg.ema_decay, cfg.reset_threshold)
            for _ in range(cfg.num_quantizers)
        )

    @property
    def units(self) -> nn.ModuleList:
        return self.layers

    def combine_codes(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(codes, dim=0).sum(0)

    def active_levels(self) -> int:
        if self.training and self.dropout_p > 0 and torch.rand(()) < self.dropout_p:
            return int(torch.randint(1, self.num_quantizers + 1, ()).item())

        return self.num_quantizers

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        commit = z.new_zeros(())
        indices: list[torch.Tensor] = []
        n_active = self.active_levels()

        for i, layer in enumerate(self.layers):
            if i < n_active:
                code, idx, c = layer(residual)
                residual = residual - code
                quantized = quantized + code
                commit = commit + c
                indices.append(idx)
            else:  # sentinel, never a valid code: a dropped level must not read as code 0
                indices.append(
                    torch.full(z.shape[:-1], DROPPED_CODE, dtype=torch.long, device=z.device)
                )

        if self.training:
            quantized = z + (quantized - z).detach()

        return quantized, torch.stack(indices, dim=-1), commit

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        out = 0.0

        for i, layer in enumerate(self.layers):
            out = out + layer.lookup(indices[..., i])

        return out


class RvqBaselineTokenizer(MotionTokenizer):
    def __init__(self, cfg: RvqBaselineCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder1d(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)
        self.decoder = Decoder1d(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)
        self.pre_q = nn.Linear(cfg.width, cfg.code_dim)
        self.post_q = nn.Linear(cfg.code_dim, cfg.width)
        self.rvq = ResidualVQ(cfg)

    @property
    def codebook_size(self) -> int:
        return self.cfg.codebook_size

    @property
    def quantizer(self) -> ResidualVQ:  # checkpoints keep the historical `rvq` submodule name
        return self.rvq

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_q(self.encoder(x))
        _, indices, _ = self.rvq(z)
        return reject_dropped_codes(indices)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.rvq.indices_to_codes(indices)
        return self.decoder(self.post_q(codes))

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        z = self.pre_q(self.encoder(x))
        quantized, indices, commit = self.rvq(z)
        recon = self.decoder(self.post_q(quantized))
        return TokenizerOutput(recon=recon, indices=indices, commit=commit)
