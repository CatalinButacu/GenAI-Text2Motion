"""Strong RVQ baseline-to-beat for Contribution A.

A residual vector-quantized VAE with the canonical EMA-codebook recipe (van den Oord EMA update,
arXiv:1711.00937 App. A.1; T2M-GPT `ema_reset` arXiv:2301.06052; MoMask residual stack +
quantization dropout arXiv:2312.00063; EnCodec/SoundStream mechanics arXiv:2210.13438). It is what
the Residual-FSQ tokenizer must beat on reconstruction + downstream FID.

It deliberately SHARES the conv encoder/decoder with the FSQ tokenizer (`Encoder1d`/`Decoder1d`), so
the only difference between the two tokenizers is the quantizer (EMA-VQ vs FSQ) at matched encoder
capacity, sequence length and loss. `encode`/`decode` return the same `(B, T', num_quantizers)` index
contract as `ResidualFsqTokenizer`, so the generator trains on either interchangeably.

Built from `RvqBaselineCfg`. See `.claude/skills/motion-tokenizer` and `.claude/docs/references.md`.
"""

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.model.tokenizer import Decoder1d, Encoder1d
from text2motion.shared.config import RvqBaselineCfg


class EmaVectorQuantizer(nn.Module):
    """Single-level VQ with EMA codebook updates + dead-code reset.

    The codebook is a non-gradient buffer updated by exponential moving averages of the encoder
    vectors assigned to each code (Laplace-smoothed). Codes whose EMA cluster size falls below
    `reset_threshold` are reinitialised to random vectors from the current batch -- the standard
    anti-collapse trick. A commitment loss pulls the encoder toward the chosen code.
    """

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
        """(N, D) -> (N, K) squared L2 to each code."""
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
        """z (B, T, D) -> (clean quantized (B, T, D), indices (B, T), commitment loss). The
        straight-through estimator is applied once at the residual-stack level, not here."""
        # Assignment + EMA codebook update are non-differentiable (argmin) and must not leak the
        # autograd graph into the in-place buffer updates -- the encoder gradient comes from the
        # straight-through `z` term and the commitment loss on `z`, not from `quant`.
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
        """indices (...) -> codes (..., D)."""
        return F.embedding(indices, self.embed)


class ResidualVQ(nn.Module):
    """Stack of EMA-VQ levels over successive residuals, with quantization dropout."""

    def __init__(self, cfg: RvqBaselineCfg) -> None:
        super().__init__()
        self.num_quantizers = cfg.num_quantizers
        self.dropout_p = cfg.quant_dropout
        self.layers = nn.ModuleList(
            EmaVectorQuantizer(cfg.codebook_size, cfg.code_dim, cfg.ema_decay, cfg.reset_threshold)
            for _ in range(cfg.num_quantizers)
        )

    def active_levels(self) -> int:
        if self.training and self.dropout_p > 0 and torch.rand(()) < self.dropout_p:
            return int(torch.randint(1, self.num_quantizers + 1, ()).item())

        return self.num_quantizers

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z (B, T, D) -> (quantized sum, indices (B, T, num_quantizers), summed commitment loss)."""
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
            else:
                indices.append(torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device))

        # Straight-through only matters for the training-time encoder gradient; in eval return the
        # true quantized so decode(encode) == forward exactly.
        if self.training:
            quantized = z + (quantized - z).detach()

        return quantized, torch.stack(indices, dim=-1), commit

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B, T, num_quantizers) -> summed codes (B, T, D)."""
        out = 0.0

        for i, layer in enumerate(self.layers):
            out = out + layer.lookup(indices[..., i])

        return out


class RvqBaselineTokenizer(nn.Module):
    """263 motion feature <-> residual-VQ tokens (the baseline). Same I/O contract as the FSQ
    tokenizer: `encode` -> indices, `decode` -> reconstruction, `forward` -> (recon, indices, commit)."""

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, in_dim) -> indices (B, T/downsample, num_quantizers)."""
        z = self.pre_q(self.encoder(x))
        _, indices, _ = self.rvq(z)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B, T', num_quantizers) -> reconstruction (B, T, in_dim)."""
        codes = self.rvq.indices_to_codes(indices)
        return self.decoder(self.post_q(codes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x (B, T, in_dim) -> (reconstruction, indices (B, T', num_quantizers), commitment loss)."""
        z = self.pre_q(self.encoder(x))
        quantized, indices, commit = self.rvq(z)
        recon = self.decoder(self.post_q(quantized))
        return recon, indices, commit
