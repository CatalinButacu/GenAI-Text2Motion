"""Residual-FSQ motion tokenizer — Contribution A.

FSQ (Finite Scalar Quantization, Mentzer et al., arXiv:2309.15505) bounds each latent dim with
tanh and rounds to fixed levels; the implicit codebook is the product of per-dim level sets, giving
~100% usage with NO collapse, NO EMA/commitment loss. A residual stack of FSQ quantizers (residual
structure proven on motion by MoMask, arXiv:2312.00063) + quantization dropout is the novel,
supported combination (FSQ-on-motion proven by ScaMo arXiv:2412.14559 and arXiv:2508.08991).

Built from `TokenizerCfg` (no magic numbers). Encoder/decoder are 1D temporal convs over the
263-dim HumanML3D feature (downsample `d`); the quantizer operates on a low-dim FSQ projection.
See `.claude/skills/motion-tokenizer` and `.claude/docs/references.md`.
"""

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.shared.config import TokenizerCfg


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient."""
    return z + (torch.round(z) - z).detach()


class FSQ(nn.Module):
    """Finite Scalar Quantization over `len(levels)` dims (Mentzer et al., arXiv:2309.15505)."""

    def __init__(self, levels: tuple[int, ...]) -> None:
        super().__init__()
        levels_t = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", levels_t)
        self.register_buffer("half_l_bound", (levels_t - 1) * 0.5)  # for the tanh bound
        self.register_buffer("half_width", torch.div(levels_t, 2, rounding_mode="floor"))  # L//2
        basis = torch.cumprod(torch.tensor([1] + list(levels[:-1]), dtype=torch.long), dim=0)
        self.register_buffer("basis", basis)
        self.dim = len(levels)
        self.codebook_size = int(torch.prod(levels_t).item())

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_l = self.half_l_bound * (1 - eps)
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """z (..., dim) -> quantized codes in normalised [-1, 1], straight-through."""
        quantized = round_ste(self.bound(z))
        return quantized / self.half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Normalised codes (..., dim) -> integer indices (...,) in [0, codebook_size)."""
        shifted = (codes * self.half_width) + self.half_width
        return (shifted.round().long() * self.basis).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Integer indices (...,) -> normalised codes (..., dim)."""
        idx = indices.unsqueeze(-1)
        levels = self.levels.long()
        digits = (idx // self.basis) % levels
        return (digits.float() - self.half_width) / self.half_width


class ResidualFSQ(nn.Module):
    """Stack of FSQ quantizers over successive residuals, with quantization dropout."""

    def __init__(self, levels: tuple[int, ...], num_quantizers: int, dropout_p: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FSQ(levels) for _ in range(num_quantizers)])
        self.num_quantizers = num_quantizers
        self.dropout_p = dropout_p

    def active_levels(self) -> int:
        """How many residual levels to apply this step (quantization dropout, training only)."""
        if self.training and self.dropout_p > 0 and torch.rand(()) < self.dropout_p:
            return int(torch.randint(1, self.num_quantizers + 1, ()).item())

        return self.num_quantizers

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (B, T, dim) -> (quantized sum (B, T, dim), indices (B, T, num_quantizers))."""
        residual = z
        quantized = torch.zeros_like(z)
        indices: list[torch.Tensor] = []
        n_active = self.active_levels()

        for i, layer in enumerate(self.layers):
            if i < n_active:
                code = layer.quantize(residual)
                residual = residual - code
                quantized = quantized + code
                indices.append(layer.codes_to_indices(code))
            else:
                indices.append(torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device))

        return quantized, torch.stack(indices, dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B, T, num_quantizers) -> summed normalised codes (B, T, dim)."""
        out = 0.0

        for i, layer in enumerate(self.layers):
            out = out + layer.indices_to_codes(indices[..., i])

        return out


class GroupedFSQ(nn.Module):
    """``num_groups`` independent FSQ groups over a ``(num_groups * dim)`` latent. Concatenating the
    per-group codes gives a ``num_groups * dim``-dimensional quantized latent — far higher capacity
    than residual FSQ (which stays at ``dim`` and whose later levels collapse on the fixed grid).
    One token per group; every group fully uses its codebook."""

    def __init__(self, levels: tuple[int, ...], num_groups: int) -> None:
        super().__init__()
        self.groups = nn.ModuleList([FSQ(levels) for _ in range(num_groups)])
        self.num_groups = num_groups
        self.dim = len(levels)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (B, T, num_groups*dim) -> (quantized (B, T, num_groups*dim), indices (B, T, num_groups))."""
        codes: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for group_index, group in enumerate(self.groups):
            chunk = z[..., group_index * self.dim : (group_index + 1) * self.dim]
            code = group.quantize(chunk)
            codes.append(code)
            indices.append(group.codes_to_indices(code))
        return torch.cat(codes, dim=-1), torch.stack(indices, dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B, T, num_groups) -> concatenated codes (B, T, num_groups*dim)."""
        return torch.cat(
            [group.indices_to_codes(indices[..., i]) for i, group in enumerate(self.groups)], dim=-1
        )


class ResBlock1d(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder1d(nn.Module):
    """(B, T, in_dim) -> (B, T/downsample, width). downsample is a power of 2 via stride-2 convs."""

    def __init__(self, in_dim: int, width: int, downsample: int, n_resblocks: int) -> None:
        super().__init__()
        n_down = downsample.bit_length() - 1  # downsample=4 -> 2 stride-2 convs
        layers: list[nn.Module] = [nn.Conv1d(in_dim, width, 3, padding=1), nn.ReLU()]

        for _ in range(n_down):
            layers += [nn.Conv1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        for _ in range(n_resblocks):
            layers.append(ResBlock1d(width))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class Decoder1d(nn.Module):
    """(B, T/downsample, width) -> (B, T, in_dim)."""

    def __init__(self, in_dim: int, width: int, downsample: int, n_resblocks: int) -> None:
        super().__init__()
        n_up = downsample.bit_length() - 1
        layers: list[nn.Module] = []

        for _ in range(n_resblocks):
            layers.append(ResBlock1d(width))

        for _ in range(n_up):
            layers += [nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        layers.append(nn.Conv1d(width, in_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ResidualFsqTokenizer(nn.Module):
    """263 motion feature <-> residual-FSQ tokens. `encode` returns indices; `decode` reconstructs."""

    def __init__(self, cfg: TokenizerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        fsq_dim = len(cfg.fsq_levels)
        self.encoder = Encoder1d(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)
        self.decoder = Decoder1d(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)

        if cfg.quantizer == "grouped":
            latent_dim = cfg.num_quantizers * fsq_dim
            self.quantizer: nn.Module = GroupedFSQ(cfg.fsq_levels, cfg.num_quantizers)
        elif cfg.quantizer == "residual":
            latent_dim = fsq_dim
            self.quantizer = ResidualFSQ(cfg.fsq_levels, cfg.num_quantizers, cfg.quant_dropout)
        else:
            raise ValueError(f"quantizer must be 'grouped' or 'residual', got {cfg.quantizer!r}")

        self.pre_q = nn.Linear(cfg.width, latent_dim)
        self.post_q = nn.Linear(latent_dim, cfg.width)

    @property
    def codebook_size(self) -> int:
        return self.quantizer.groups[0].codebook_size if self.cfg.quantizer == "grouped" \
            else self.quantizer.layers[0].codebook_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, in_dim) -> indices (B, T/downsample, num_quantizers)."""
        _, indices = self.quantizer(self.pre_q(self.encoder(x)))
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B, T', num_quantizers) -> reconstruction (B, T, in_dim)."""
        return self.decoder(self.post_q(self.quantizer.indices_to_codes(indices)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, T, in_dim) -> (reconstruction (B, T, in_dim), indices (B, T', num_quantizers))."""
        quantized, indices = self.quantizer(self.pre_q(self.encoder(x)))
        recon = self.decoder(self.post_q(quantized))
        return recon, indices


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on the feature + L1 on its temporal velocity (smoothness). FSQ needs no commitment term."""
    feat = F.l1_loss(recon, target)
    vel = F.l1_loss(recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1])

    return feat + vel
