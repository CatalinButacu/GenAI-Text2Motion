from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.motion.dataset import MotionScaler
from text2motion.motion.model import MotionClip, MotionTokens
from text2motion.motion.representation import DIM


class Quantizer(StrEnum):
    GROUPED = "grouped"
    RESIDUAL = "residual"


class TokenizerKind(StrEnum):
    FSQ = "fsq"
    RVQ = "rvq"


@dataclass(frozen=True)
class TokenizerConfig:
    kind: TokenizerKind = TokenizerKind.FSQ
    in_dim: int = DIM
    width: int = 512
    downsample: int = 4
    num_quantizers: int = 6
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    quantizer: Quantizer = Quantizer.GROUPED
    quant_dropout: float = 0.2
    n_resblocks: int = 3


@dataclass(frozen=True)
class RvqConfig:
    in_dim: int = DIM
    width: int = 512
    downsample: int = 4
    n_resblocks: int = 3
    num_quantizers: int = 6
    codebook_size: int = 512
    code_dim: int = 512
    ema_decay: float = 0.99
    commitment_beta: float = 0.02
    quant_dropout: float = 0.2
    reset_threshold: float = 1.0


DROPPED_CODE = -1


class TokenizerOutput(NamedTuple):
    recon: torch.Tensor
    indices: torch.Tensor
    commit: torch.Tensor


class MotionQuantizer(nn.Module, ABC):
    @property
    @abstractmethod
    def units(self) -> nn.ModuleList: ...

    @abstractmethod
    def combine_codes(self, codes: list[torch.Tensor]) -> torch.Tensor: ...

    @abstractmethod
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor: ...


class TokenizerModule(nn.Module, ABC):
    @property
    @abstractmethod
    def codebook_size(self) -> int: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, indices: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> TokenizerOutput: ...


def reject_dropped_codes(indices: torch.Tensor) -> torch.Tensor:
    if bool((indices < 0).any()):
        raise RuntimeError(
            "quantizer dropout left inactive levels in this batch, so the emitted indices do not "
            "describe the motion. Call encode() in eval mode, or disable quant_dropout, before "
            "tokenizing a corpus or training a generator on these tokens."
        )
    return indices


def round_ste(z: torch.Tensor) -> torch.Tensor:
    return z + (torch.round(z) - z).detach()


class FSQ(nn.Module):
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


class ResidualFSQ(MotionQuantizer):
    def __init__(self, levels: tuple[int, ...], num_quantizers: int, dropout_p: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FSQ(levels) for _ in range(num_quantizers)])
        self.num_quantizers = num_quantizers
        self.dropout_p = dropout_p

    @property
    def units(self) -> nn.ModuleList:
        return self.layers

    def combine_codes(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(codes, dim=0).sum(0)

    def active_levels(self) -> int:
        if self.training and self.dropout_p > 0 and torch.rand(()) < self.dropout_p:
            return int(torch.randint(1, self.num_quantizers + 1, ()).item())

        return self.num_quantizers

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
                indices.append(
                    torch.full(z.shape[:-1], DROPPED_CODE, dtype=torch.long, device=z.device)
                )

        return quantized, torch.stack(indices, dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        out = 0.0

        for i, layer in enumerate(self.layers):
            out = out + layer.indices_to_codes(indices[..., i])

        return out


class GroupedFSQ(MotionQuantizer):
    def __init__(self, levels: tuple[int, ...], num_quantizers: int) -> None:
        super().__init__()
        self.groups = nn.ModuleList([FSQ(levels) for _ in range(num_quantizers)])
        self.num_quantizers = num_quantizers
        self.dim = len(levels)

    @property
    def units(self) -> nn.ModuleList:
        return self.groups

    def combine_codes(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(codes, dim=-1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for group_index, group in enumerate(self.groups):
            chunk = z[..., group_index * self.dim : (group_index + 1) * self.dim]
            code = group.quantize(chunk)
            codes.append(code)
            indices.append(group.codes_to_indices(code))
        return torch.cat(codes, dim=-1), torch.stack(indices, dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [group.indices_to_codes(indices[..., i]) for i, group in enumerate(self.groups)], dim=-1
        )


def resample_stages(downsample: int) -> int:
    if downsample < 1 or downsample & (downsample - 1):
        raise ValueError(
            f"tokenizer downsample must be a power of two, got {downsample}; the encoder builds "
            f"stride-2 stages and any other value would silently change the frame rate per token"
        )
    return downsample.bit_length() - 1


class ResBlock1d(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder1d(nn.Module):
    def __init__(self, in_dim: int, width: int, downsample: int, n_resblocks: int) -> None:
        super().__init__()
        n_down = resample_stages(downsample)
        layers: list[nn.Module] = [nn.Conv1d(in_dim, width, 3, padding=1), nn.ReLU()]

        for _ in range(n_down):
            layers += [nn.Conv1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        for _ in range(n_resblocks):
            layers.append(ResBlock1d(width))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class Decoder1d(nn.Module):
    def __init__(self, in_dim: int, width: int, downsample: int, n_resblocks: int) -> None:
        super().__init__()
        n_up = resample_stages(downsample)
        layers: list[nn.Module] = []

        for _ in range(n_resblocks):
            layers.append(ResBlock1d(width))

        for _ in range(n_up):
            layers += [nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        layers.append(nn.Conv1d(width, in_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def _build_grouped_fsq(cfg: TokenizerConfig) -> tuple[nn.Module, int]:
    return GroupedFSQ(cfg.fsq_levels, cfg.num_quantizers), cfg.num_quantizers * len(cfg.fsq_levels)


def _build_residual_fsq(cfg: TokenizerConfig) -> tuple[nn.Module, int]:
    quantizer = ResidualFSQ(cfg.fsq_levels, cfg.num_quantizers, cfg.quant_dropout)
    return quantizer, len(cfg.fsq_levels)


QUANTIZER_FACTORIES = {
    Quantizer.GROUPED: _build_grouped_fsq,
    Quantizer.RESIDUAL: _build_residual_fsq,
}


class ResidualFsqTokenizer(TokenizerModule):
    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder1d(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)
        self.decoder = Decoder1d(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)

        build = QUANTIZER_FACTORIES.get(Quantizer(cfg.quantizer))
        if build is None:
            raise ValueError(
                f"quantizer must be one of {[q.value for q in Quantizer]}, got {cfg.quantizer!r}"
            )
        self.quantizer, latent_dim = build(cfg)

        self.pre_q = nn.Linear(cfg.width, latent_dim)
        self.post_q = nn.Linear(latent_dim, cfg.width)

    @property
    def codebook_size(self) -> int:
        return self.quantizer.units[0].codebook_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, indices = self.quantizer(self.pre_q(self.encoder(x)))
        return reject_dropped_codes(indices)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_q(self.quantizer.indices_to_codes(indices)))

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        quantized, indices = self.quantizer(self.pre_q(self.encoder(x)))
        recon = self.decoder(self.post_q(quantized))
        return TokenizerOutput(recon=recon, indices=indices, commit=x.new_zeros(()))


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    feat = F.l1_loss(recon, target)
    vel = F.l1_loss(recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1])

    return feat + vel


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
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        return (
            flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1)
        )

    def _ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        n = onehot.sum(0)
        embed_sum = onehot.t() @ flat
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
            indices = self._distances(flat).argmin(1)
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
    def __init__(self, cfg: RvqConfig) -> None:
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
            else:
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


class RvqBaselineTokenizer(TokenizerModule):
    def __init__(self, cfg: RvqConfig) -> None:
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
    def quantizer(self) -> ResidualVQ:
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


TOKENIZER_FACTORIES = {
    TokenizerKind.FSQ: lambda fsq, rvq: ResidualFsqTokenizer(fsq),
    TokenizerKind.RVQ: lambda fsq, rvq: RvqBaselineTokenizer(rvq),
}


def build_tokenizer_module(fsq: TokenizerConfig, rvq: RvqConfig | None = None) -> TokenizerModule:
    kind = TokenizerKind(fsq.kind)
    build = TOKENIZER_FACTORIES.get(kind)
    if build is None:
        raise ValueError(
            f"tokenizer kind must be one of {[k.value for k in TokenizerKind]}, got {fsq.kind!r}"
        )
    return build(fsq, rvq or RvqConfig())


class MotionTokenizer:
    def __init__(
        self,
        module: TokenizerModule,
        downsample: int,
        device: str | torch.device = "cpu",
        scaler: MotionScaler | None = None,
    ) -> None:
        self.module = module.to(device).eval()
        self.downsample = downsample
        self.device = device
        self.scaler = scaler

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        fsq: TokenizerConfig,
        rvq: RvqConfig | None = None,
        device: str | torch.device = "cpu",
        scaler: MotionScaler | None = None,
    ) -> MotionTokenizer:
        module = build_tokenizer_module(fsq, rvq)
        module.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(module, downsample=fsq.downsample, device=device, scaler=scaler)

    @property
    def codebook_size(self) -> int:
        return int(self.module.codebook_size)

    @property
    def num_codebooks(self) -> int:
        return len(self.module.quantizer.units)

    def usable_frames(self, frame_count: int) -> int:
        return (frame_count // self.downsample) * self.downsample

    @torch.no_grad()
    def encode(self, motion: MotionClip) -> MotionTokens:
        usable = self.usable_frames(motion.frame_count)
        if usable == 0:
            raise ValueError(
                f"clip has {motion.frame_count} frames, fewer than one token at downsample "
                f"{self.downsample}"
            )
        features = motion.features[:usable].unsqueeze(0).to(self.device)
        indices = self.module.encode(features)[0]
        return MotionTokens(
            indices=indices,
            token_count=int(indices.shape[0]),
            frame_count=usable,
        )

    @torch.no_grad()
    def decode(self, tokens: MotionTokens) -> MotionClip:
        indices = tokens.indices
        if indices.dim() == 2:
            indices = indices.unsqueeze(0)
        features = self.module.decode(indices.to(self.device))[0]
        return MotionClip(features=features, frame_count=int(features.shape[0]))
