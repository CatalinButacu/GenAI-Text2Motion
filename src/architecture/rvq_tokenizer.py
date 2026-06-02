"""Residual Vector-Quantized tokenizer for SMPL-X motion.

Encodes (T, 168) motion -> K discrete codebook indices per latent frame, and decodes back.
Architecture follows MoMask (arXiv 2312.00063) and Mogo (arXiv 2412.07797):
    encoder: Conv1D stride-2 stack (down_t total downsample)
    rvq:     K residual codebooks, V entries each, latent_dim wide
    decoder: ConvTranspose1D stack OR causal (CausalConv1d + nearest upsample)
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.constants import CONSTS, RVQ, SMPLX

log = logging.getLogger(__name__)

def stride_count(down_t: int) -> int:
    """Number of stride-2 blocks needed for total downsample factor down_t."""
    return int(math.log2(down_t))


class RVQCodebook(nn.Module):
    """Single VQ codebook with EMA-updated entries (no straight-through codebook loss)."""

    codebook: torch.Tensor
    cluster_size: torch.Tensor
    embed_avg: torch.Tensor

    def __init__(
        self,
        num_entries: int,
        latent_dim: int,
        decay: float = RVQ.ema_decay,
        eps: float = RVQ.ema_eps,
    ):
        super().__init__()
        self.num_entries = num_entries
        self.latent_dim = latent_dim
        self.decay = decay
        self.eps = eps

        codebook = torch.randn(num_entries, latent_dim) * RVQ.codebook_init_scale
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(num_entries))
        self.register_buffer("embed_avg", codebook.clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, self.latent_dim)

        dists = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )
        indices = dists.argmin(dim=1)
        onehot = F.one_hot(indices, self.num_entries).type(flat.dtype)

        if self.training:
            self.ema_update(flat, onehot)

        quantized = self.codebook[indices].reshape_as(x)
        # Straight-through: gradients flow from quantized back to x (encoder).
        quantized_st = x + (quantized - x).detach()
        indices = indices.reshape(x.shape[:-1])

        return quantized_st, indices, quantized

    @torch.no_grad()
    def ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        n = onehot.sum(dim=0)
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)

        embed_sum = onehot.t() @ flat
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + self.num_entries * self.eps) * total
        self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))

    @torch.no_grad()
    def reset_dead_codes(
        self,
        flat: torch.Tensor,
        threshold: float = RVQ.dead_code_reset_threshold,
        noise_std: float = RVQ.dead_code_noise_std,
    ) -> int:
        """VQ-VAE-2 revival: replace entries with cluster_size<threshold with batch samples."""
        dead = (self.cluster_size < threshold).nonzero(as_tuple=True)[0]

        if dead.numel() == 0 or flat.numel() == 0:
            return 0
        n = dead.numel()
        sample_idx = torch.randint(0, flat.shape[0], (n,), device=flat.device)
        replacements = flat[sample_idx] + noise_std * torch.randn_like(flat[sample_idx])
        self.codebook[dead] = replacements
        self.embed_avg[dead] = replacements
        self.cluster_size[dead] = RVQ.dead_code_reset_threshold

        return int(n)

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        # Guard against out-of-range from a misconfigured sampler — silent wrap is the bug class.
        if indices.numel() > 0:
            mx = int(indices.max().item())
            mn = int(indices.min().item())
            assert 0 <= mn and mx < self.num_entries, (
                f"codebook index out of range: [{mn}, {mx}] vs [0, {self.num_entries})"
            )
        return self.codebook[indices]


class ResidualVectorQuantizer(nn.Module):
    """Stack of K VQ codebooks. Each quantizes the residual of the previous."""

    def __init__(self, n_codebooks: int, codebook_size: int, latent_dim: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.codebooks = nn.ModuleList(
            [RVQCodebook(codebook_size, latent_dim) for _ in range(n_codebooks)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        quantized_sum = torch.zeros_like(x)
        all_indices = []
        commit_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        for cb in self.codebooks:
            q_st, idx, q_hard = cb(residual)
            quantized_sum = quantized_sum + q_st
            # Two-term commitment (Oord et al. 2017 eq. 4): sg[z]<->e + z<->sg[e].
            commit_loss = (
                commit_loss
                + F.mse_loss(residual.detach(), q_hard)
                + F.mse_loss(residual, q_hard.detach())
            )
            residual = residual - q_st
            all_indices.append(idx)

        indices = torch.stack(all_indices, dim=-1)
        commit_loss = commit_loss / self.n_codebooks

        return quantized_sum, indices, commit_loss

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        first_cb: RVQCodebook = self.codebooks[0]  # type: ignore[assignment]
        out = torch.zeros(
            (*indices.shape[:-1], self.latent_dim),
            device=indices.device,
            dtype=first_cb.codebook.dtype,
        )
        for k, module in enumerate(self.codebooks):
            cb: RVQCodebook = module  # type: ignore[assignment]
            out = out + cb.decode_indices(indices[..., k])

        return out

    @torch.no_grad()
    def reset_dead_codes_pipeline(
        self, x: torch.Tensor, threshold: float = RVQ.dead_code_reset_threshold
    ) -> list[int]:
        """Walk the residual chain like forward, resetting dead codes per layer."""
        flat = x.reshape(-1, self.latent_dim)
        counts: list[int] = []

        for module in self.codebooks:
            cb: RVQCodebook = module  # type: ignore[assignment]
            n_reset = cb.reset_dead_codes(flat, threshold)
            counts.append(n_reset)
            dists = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat @ cb.codebook.t()
                + cb.codebook.pow(2).sum(dim=1)
            )
            idx = dists.argmin(dim=1)
            flat = flat - cb.codebook[idx]

        return counts


class CausalConv1d(nn.Conv1d):
    """1D conv with left-only padding so output[t] depends on input[:t+1] only."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=0)
        self.left_pad = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(F.pad(x, (self.left_pad, 0)))


class CausalUpsample2x(nn.Module):
    """Nearest-neighbor 2x upsample (causal-safe: each output inherits from one past input)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest")


class MotionRVQTokenizer(nn.Module):
    """End-to-end motion VQ-VAE: encode 168-d pose sequences to K-codebook indices.
    causal_decoder=False (default) uses ConvTranspose1d so existing checkpoints load unchanged.
    causal_decoder=True swaps in CausalConv1d + nearest-upsample for streaming inference."""

    def __init__(
        self,
        motion_dim: int = SMPLX.pose_dim,
        latent_dim: int = 128,
        n_codebooks: int = 6,
        codebook_size: int = 512,
        down_t: int = 4,
        causal_decoder: bool = False,
    ):
        super().__init__()
        assert down_t in RVQ.valid_down_t, f"down_t must be one of {RVQ.valid_down_t}"
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.down_t = down_t
        self.causal_decoder = causal_decoder

        self.encoder = self.build_encoder(motion_dim, latent_dim, down_t)
        self.rvq = ResidualVectorQuantizer(n_codebooks, codebook_size, latent_dim)

        if causal_decoder:
            self.decoder = self.build_causal_decoder(latent_dim, motion_dim, down_t)
        else:
            self.decoder = self.build_decoder(latent_dim, motion_dim, down_t)

    @staticmethod
    def build_encoder(in_dim: int, latent_dim: int, down_t: int) -> nn.Sequential:
        n_stride = stride_count(down_t)
        hidden = latent_dim
        layers: list[nn.Module] = [
            nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(n_stride):
            layers += [
                nn.Conv1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
            ]
        layers.append(nn.Conv1d(hidden, latent_dim, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    @staticmethod
    def build_decoder(latent_dim: int, out_dim: int, down_t: int) -> nn.Sequential:
        n_stride = stride_count(down_t)
        hidden = latent_dim
        layers: list[nn.Module] = [
            nn.Conv1d(latent_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(n_stride):
            layers += [
                nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
            ]
        layers.append(nn.Conv1d(hidden, out_dim, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    @staticmethod
    def build_causal_decoder(latent_dim: int, out_dim: int, down_t: int) -> nn.Sequential:
        """Causal counterpart of build_decoder: CausalConv1d + nearest upsample preserves
        causality (output frame t depends only on latents 0..ceil(t / down_t))."""
        n_stride = stride_count(down_t)
        hidden = latent_dim
        layers: list[nn.Module] = [
            CausalConv1d(latent_dim, hidden, kernel_size=3),
            nn.SiLU(),
        ]

        for _ in range(n_stride):
            layers += [
                CausalUpsample2x(),
                CausalConv1d(hidden, hidden, kernel_size=4),
                nn.SiLU(),
                CausalConv1d(hidden, hidden, kernel_size=3),
                nn.SiLU(),
            ]
        layers.append(CausalConv1d(hidden, out_dim, kernel_size=3))

        return nn.Sequential(*layers)

    def codebook_utilization(self) -> list[dict]:
        """Per-codebook entropy + active fraction; call post-epoch to detect collapse."""
        results = []
        for module in self.rvq.codebooks:
            cb: RVQCodebook = module  # type: ignore[assignment]
            sizes = cb.cluster_size.float()
            total = sizes.sum().clamp(min=1.0)
            p = sizes / total
            nonzero = p[p > 0]
            entropy = -(nonzero * nonzero.log2()).sum().item() if len(nonzero) > 0 else 0.0
            max_entropy = math.log2(cb.num_entries)
            active = (sizes > RVQ.active_code_threshold).float().mean().item()
            results.append(
                {
                    "entropy": entropy,
                    "max_entropy": max_entropy,
                    "active_fraction": active,
                }
            )
        return results

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        x = motion.transpose(1, 2)
        z = self.encoder(x).transpose(1, 2)
        _, indices, _ = self.rvq(z)

        return indices

    @torch.no_grad()
    def reset_dead_codes(
        self, motion: torch.Tensor, threshold: float = RVQ.dead_code_reset_threshold
    ) -> list[int]:
        was_training = self.training
        self.eval()
        x = motion.transpose(1, 2)
        z = self.encoder(x).transpose(1, 2)
        counts = self.rvq.reset_dead_codes_pipeline(z, threshold)

        if was_training:
            self.train()

        return counts

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        z = self.rvq.decode(indices).transpose(1, 2)

        return self.decoder(z).transpose(1, 2)

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = motion.transpose(1, 2)
        z = self.encoder(x).transpose(1, 2)
        quantized, indices, commit_loss = self.rvq(z)
        recon = self.decoder(quantized.transpose(1, 2)).transpose(1, 2)

        return recon, indices, commit_loss
