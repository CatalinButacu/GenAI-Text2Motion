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
    clusterSize: torch.Tensor
    embedAvg: torch.Tensor

    def __init__(self, numEntries: int, latentDim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.numEntries = numEntries
        self.latentDim = latentDim
        self.decay = decay
        self.eps = eps

        codebook = torch.randn(numEntries, latentDim) * 0.01
        self.register_buffer("codebook", codebook)
        self.register_buffer("clusterSize", torch.zeros(numEntries))
        self.register_buffer("embedAvg", codebook.clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, D) -- continuous latent to quantize
        flat = x.reshape(-1, self.latentDim)  # (B*T, D)

        # Squared euclidean distance to every codebook entry
        dists = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )
        indices = dists.argmin(dim=1)  # (B*T,)
        onehot = F.one_hot(indices, self.numEntries).type(flat.dtype)

        if self.training:
            self.emaUpdate(flat, onehot)

        quantized = self.codebook[indices].reshape_as(x)
        # Straight-through: gradients flow from quantized back to x (encoder)
        quantizedSt = x + (quantized - x).detach()
        indices = indices.reshape(x.shape[:-1])

        return quantizedSt, indices, quantized

    @torch.no_grad()
    def emaUpdate(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        n = onehot.sum(dim=0)
        self.clusterSize.mul_(self.decay).add_(n, alpha=1 - self.decay)

        embedSum = onehot.t() @ flat
        self.embedAvg.mul_(self.decay).add_(embedSum, alpha=1 - self.decay)

        total = self.clusterSize.sum()
        smoothed = (self.clusterSize + self.eps) / (total + self.numEntries * self.eps) * total
        self.codebook.copy_(self.embedAvg / smoothed.unsqueeze(1))

    @torch.no_grad()
    def decodeIndices(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, T) -> (B, T, D)
        return self.codebook[indices]


class ResidualVectorQuantizer(nn.Module):
    """Stack of K VQ codebooks. Each quantizes the residual of the previous."""

    def __init__(self, nCodebooks: int, codebookSize: int, latentDim: int):
        super().__init__()
        self.nCodebooks = nCodebooks
        self.codebookSize = codebookSize
        self.latentDim = latentDim
        self.codebooks = nn.ModuleList(
            [RVQCodebook(codebookSize, latentDim) for _ in range(nCodebooks)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, D) -> quantized (B, T, D), indices (B, T, K), commit_loss scalar
        residual = x
        quantizedSum = torch.zeros_like(x)
        allIndices = []
        commitLoss = torch.zeros((), device=x.device, dtype=x.dtype)

        for cb in self.codebooks:
            q_st, idx, q_hard = cb(residual)
            quantizedSum = quantizedSum + q_st
            # Two-term commitment loss (Oord et al. 2017, eq. 4):
            #   term 1 (sg[z] - e)^2  — moves codebook toward encoder output
            #   term 2 β*(z - sg[e])^2 — encoder commitment (β handled externally)
            commitLoss = commitLoss + F.mse_loss(residual.detach(), q_hard) + F.mse_loss(
                residual, q_hard.detach()
            )
            residual = residual - q_st
            allIndices.append(idx)

        indices = torch.stack(allIndices, dim=-1)  # (B, T, K)
        commitLoss = commitLoss / self.nCodebooks

        return quantizedSum, indices, commitLoss

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, T, K) -> (B, T, D)
        firstCb: RVQCodebook = self.codebooks[0]  # type: ignore[assignment]
        out = torch.zeros(
            (*indices.shape[:-1], self.latentDim),
            device=indices.device,
            dtype=firstCb.codebook.dtype,
        )
        for k, module in enumerate(self.codebooks):
            cb: RVQCodebook = module  # type: ignore[assignment]
            out = out + cb.decodeIndices(indices[..., k])

        return out


class MotionRVQTokenizer(nn.Module):
    """End-to-end motion VQ-VAE: encode 168-d pose sequences to K-codebook indices."""

    def __init__(
        self,
        motionDim: int = 168,
        latentDim: int = 128,
        nCodebooks: int = 6,
        codebookSize: int = 512,
        downT: int = 4,
    ):
        super().__init__()
        assert downT in (1, 2, 4, 8), "downT must be power of 2"
        self.motionDim = motionDim
        self.latentDim = latentDim
        self.downT = downT

        self.encoder = self.buildEncoder(motionDim, latentDim, downT)
        self.rvq = ResidualVectorQuantizer(nCodebooks, codebookSize, latentDim)
        self.decoder = self.buildDecoder(latentDim, motionDim, downT)

    @staticmethod
    def buildEncoder(inDim: int, latentDim: int, downT: int) -> nn.Sequential:
        # downT is applied via stride-2 blocks
        nStride = {1: 0, 2: 1, 4: 2, 8: 3}[downT]
        hidden = latentDim
        layers: list[nn.Module] = [
            nn.Conv1d(inDim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(nStride):
            layers += [
                nn.Conv1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
            ]
        layers.append(nn.Conv1d(hidden, latentDim, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    @staticmethod
    def buildDecoder(latentDim: int, outDim: int, downT: int) -> nn.Sequential:
        nStride = {1: 0, 2: 1, 4: 2, 8: 3}[downT]
        hidden = latentDim
        layers: list[nn.Module] = [
            nn.Conv1d(latentDim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(nStride):
            layers += [
                nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
            ]
        layers.append(nn.Conv1d(hidden, outDim, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    def codebookUtilization(self) -> list[dict]:
        """Per-codebook utilization: entropy (bits) and active fraction.

        Call after a training epoch to detect codebook collapse early.
        Returns list of dicts with keys: entropy, max_entropy, active_fraction.
        """
        results = []
        for module in self.rvq.codebooks:
            cb: RVQCodebook = module  # type: ignore[assignment]
            sizes = cb.clusterSize.float()
            total = sizes.sum().clamp(min=1.0)
            p = sizes / total
            nonzero = p[p > 0]
            entropy = -(nonzero * nonzero.log2()).sum().item() if len(nonzero) > 0 else 0.0
            maxEntropy = math.log2(cb.numEntries)
            active = (sizes > 0.5).float().mean().item()
            results.append({
                "entropy": entropy, "max_entropy": maxEntropy, "active_fraction": active,
            })
        return results

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        # motion: (B, T, D) -> indices (B, T', K)
        x = motion.transpose(1, 2)  # (B, D, T)
        z = self.encoder(x).transpose(1, 2)  # (B, T', latent_dim)
        _, indices, _ = self.rvq(z)

        return indices

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
