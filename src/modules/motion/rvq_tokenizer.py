"""Residual Vector-Quantized tokenizer for SMPL-X motion.

Encodes a (T, 168) motion sequence into K discrete codebook indices per latent
frame, and decodes them back. Used by TextToMotionSSM as its output head --
the transformer predicts codebook indices rather than regressing raw 168-d poses.

Architecture follows MoMask (CVPR 2024, arXiv 2312.00063) and Mogo
(arXiv 2412.07797):
    encoder: Conv1D(168, 128, stride=2) x 2  -- 4x temporal downsample
    rvq:     K residual codebooks, each V entries of 128-d
    decoder: ConvTranspose1D(128, 168, stride=2) x 2  -- 4x upsample
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class RVQCodebook(nn.Module):
    """Single VQ codebook with EMA-updated entries.

    EMA updates avoid the gradient-based codebook loss (straight-through estimator
    noise) and converge more reliably on motion data.
    """

    codebook: torch.Tensor
    cluster_size: torch.Tensor
    embed_avg: torch.Tensor

    def __init__(self, num_entries: int, latent_dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_entries = num_entries
        self.latent_dim = latent_dim
        self.decay = decay
        self.eps = eps

        codebook = torch.randn(num_entries, latent_dim) * 0.01
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(num_entries))
        self.register_buffer("embed_avg", codebook.clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, D) -- continuous latent to quantize
        flat = x.reshape(-1, self.latent_dim)  # (B*T, D)

        # Squared euclidean distance to every codebook entry
        dists = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )
        indices = dists.argmin(dim=1)  # (B*T,)
        onehot = F.one_hot(indices, self.num_entries).type(flat.dtype)

        if self.training:
            self.ema_update(flat, onehot)

        quantized = self.codebook[indices].reshape_as(x)
        # Straight-through: gradients flow from quantized back to x (encoder)
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
    def reset_dead_codes(self, flat: torch.Tensor, threshold: float = 1.0,
                       noise_std: float = 0.01) -> int:
        # VQ-VAE-2 dead-code revival: replace entries with cluster_size below
        # threshold by random latents from the current batch (+ small noise).
        dead = (self.cluster_size < threshold).nonzero(as_tuple=True)[0]

        if dead.numel() == 0 or flat.numel() == 0:
            return 0
        n = dead.numel()
        sample_idx = torch.randint(0, flat.shape[0], (n,), device=flat.device)
        replacements = flat[sample_idx] + noise_std * torch.randn_like(flat[sample_idx])
        self.codebook[dead] = replacements
        self.embed_avg[dead] = replacements
        self.cluster_size[dead] = 1.0

        return int(n)

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, T) -> (B, T, D)
        # Guard against out-of-range indices from a misconfigured sampler --
        # silently wrapping into the wrong codebook entry is the bug class
        # this assertion catches.
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
        # x: (B, T, D) -> quantized (B, T, D), indices (B, T, K), commit_loss scalar
        residual = x
        quantized_sum = torch.zeros_like(x)
        all_indices = []
        commit_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        for cb in self.codebooks:
            q_st, idx, q_hard = cb(residual)
            quantized_sum = quantized_sum + q_st
            # Two-term commitment loss (Oord et al. 2017, eq. 4):
            #   term 1 (sg[z] - e)^2  — moves codebook toward encoder output
            #   term 2 β*(z - sg[e])^2 — encoder commitment (β handled externally)
            commit_loss = commit_loss + F.mse_loss(residual.detach(), q_hard) + F.mse_loss(
                residual, q_hard.detach()
            )
            residual = residual - q_st
            all_indices.append(idx)

        indices = torch.stack(all_indices, dim=-1)  # (B, T, K)
        commit_loss = commit_loss / self.n_codebooks

        return quantized_sum, indices, commit_loss

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, T, K) -> (B, T, D)
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
    def reset_dead_codes_pipeline(self, x: torch.Tensor, threshold: float = 1.0) -> list[int]:
        # Walk the residual chain just like forward, but reset dead codes per layer
        # using whichever residual that codebook actually sees.
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
    """1D conv with left-only padding so output[t] depends on input[:t+1] only.

    Drop-in replacement for ``nn.Conv1d(..., padding='same')`` in causal decoders.
    Preserves the temporal length (input T == output T) without any future-frame
    leakage. The cost is one ``F.pad`` per layer; negligible vs the conv itself.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=0)
        self._left_pad = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(F.pad(x, (self._left_pad, 0)))


class CausalUpsample2x(nn.Module):
    """Nearest-neighbor 2x upsample. Causal-safe because every output frame
    inherits from a single past input frame; no future leakage."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest")


class MotionRVQTokenizer(nn.Module):
    """End-to-end motion VQ-VAE: encode 168-d pose sequences to K-codebook indices.

    The decoder defaults to the symmetric-padded variant (``causal_decoder=False``)
    so existing checkpoints load unchanged. Streaming inference requires a causal
    decoder (``causal_decoder=True``) which uses :class:`CausalConv1d` plus
    nearest-neighbor upsampling so output frame ``t`` only depends on latent
    tokens up to ``ceil(t / down_t)``. Training the decoder with the causal flag
    requires a fresh decoder run (encoder + codebooks can stay frozen) since
    the receptive-field alignment changes.
    """

    def __init__(
        self,
        motion_dim: int = 168,
        latent_dim: int = 128,
        n_codebooks: int = 6,
        codebook_size: int = 512,
        down_t: int = 4,
        causal_decoder: bool = False,
    ):
        super().__init__()
        assert down_t in (1, 2, 4, 8), "down_t must be power of 2"
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
        # down_t is applied via stride-2 blocks
        n_stride = {1: 0, 2: 1, 4: 2, 8: 3}[down_t]
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
        n_stride = {1: 0, 2: 1, 4: 2, 8: 3}[down_t]
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
        """Causal counterpart of :meth:`build_decoder`.

        Replaces ``ConvTranspose1d`` (which mixes future frames when its kernel
        spans both sides of the upsample boundary) with nearest-neighbor
        upsampling followed by a causally-padded ``CausalConv1d``. Same channel
        widths, same depth, same receptive-field count -- but output frame ``t``
        depends only on latent tokens ``0..ceil(t / down_t)``.

        Verified by ``tests/test_causal_rvq_decoder.py``: zeroing latents
        past index ``t / down_t`` must not change output frames ``0..t``.
        """
        n_stride = {1: 0, 2: 1, 4: 2, 8: 3}[down_t]
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
        """Per-codebook utilization: entropy (bits) and active fraction.

        Call after a training epoch to detect codebook collapse early.
        Returns list of dicts with keys: entropy, max_entropy, active_fraction.
        """
        results = []
        for module in self.rvq.codebooks:
            cb: RVQCodebook = module  # type: ignore[assignment]
            sizes = cb.cluster_size.float()
            total = sizes.sum().clamp(min=1.0)
            p = sizes / total
            nonzero = p[p > 0]
            entropy = -(nonzero * nonzero.log2()).sum().item() if len(nonzero) > 0 else 0.0
            max_entropy = math.log2(cb.num_entries)
            active = (sizes > 0.5).float().mean().item()
            results.append({
                "entropy": entropy, "max_entropy": max_entropy, "active_fraction": active,
            })
        return results

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        # motion: (B, T, D) -> indices (B, T', K)
        x = motion.transpose(1, 2)  # (B, D, T)
        z = self.encoder(x).transpose(1, 2)  # (B, T', latent_dim)
        _, indices, _ = self.rvq(z)

        return indices

    @torch.no_grad()
    def reset_dead_codes(self, motion: torch.Tensor, threshold: float = 1.0) -> list[int]:
        was_training = self.training
        self.eval()
        x = motion.transpose(1, 2)
        z = self.encoder(x).transpose(1, 2)
        counts = self.rvq.reset_dead_codes_pipeline(z, threshold)

        if was_training:
            self.train()

        return counts

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, T', K) -> motion (B, T, D)
        z = self.rvq.decode(indices).transpose(1, 2)  # (B, latent_dim, T')

        return self.decoder(z).transpose(1, 2)

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Used during tokenizer training: reconstruct + return commit loss
        x = motion.transpose(1, 2)
        z = self.encoder(x).transpose(1, 2)
        quantized, indices, commit_loss = self.rvq(z)
        recon = self.decoder(quantized.transpose(1, 2)).transpose(1, 2)

        return recon, indices, commit_loss
