from __future__ import annotations

from abc import ABC, abstractmethod
from typing import assert_never

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.motion.model import MotionClip, MotionTokens
from text2motion.tokenization.contracts import (
    FsqComposition,
    RvqConfig,
    TokenizerConfig,
    TokenizerForwardOutput,
    TokenizerKind,
)
from text2motion.tokenization.nn_quantizers import (
    GroupedFiniteScalarQuantizer,
    QuantizerStack,
    ResidualFiniteScalarQuantizer,
    ResidualVectorQuantizer,
    reject_dropped_codes,
)


class MotionTokenizerNetwork(nn.Module, ABC):
    encoder: nn.Module
    decoder: nn.Module
    pre_q: nn.Module
    post_q: nn.Module

    def latent_quantizer(self) -> QuantizerStack:
        quantizer = self.quantizer
        if not isinstance(quantizer, QuantizerStack):
            raise TypeError("the tokenizer network has no latent quantizer")
        return quantizer

    @property
    @abstractmethod
    def codebook_size(self) -> int: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, indices: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> TokenizerForwardOutput: ...


def resample_stages(downsample: int) -> int:
    if downsample < 1 or downsample & (downsample - 1):
        raise ValueError(
            f"tokenizer downsample must be a power of two, got {downsample}; the encoder builds "
            f"stride-2 stages and any other value would silently change the frame rate per token"
        )
    return downsample.bit_length() - 1


class TemporalResidualBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MotionFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, width: int, downsample: int, n_resblocks: int) -> None:
        super().__init__()
        n_down = resample_stages(downsample)
        layers: list[nn.Module] = [nn.Conv1d(in_dim, width, 3, padding=1), nn.ReLU()]

        for _ in range(n_down):
            layers += [nn.Conv1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        for _ in range(n_resblocks):
            layers.append(TemporalResidualBlock(width))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class MotionFeatureDecoder(nn.Module):
    def __init__(self, in_dim: int, width: int, downsample: int, n_resblocks: int) -> None:
        super().__init__()
        n_up = resample_stages(downsample)
        layers: list[nn.Module] = []

        for _ in range(n_resblocks):
            layers.append(TemporalResidualBlock(width))

        for _ in range(n_up):
            layers += [nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        layers.append(nn.Conv1d(width, in_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class FsqTokenizer(MotionTokenizerNetwork):
    quantizer: QuantizerStack

    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = MotionFeatureEncoder(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)
        self.decoder = MotionFeatureDecoder(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)

        match FsqComposition(cfg.quantizer):
            case FsqComposition.GROUPED:
                self.quantizer = GroupedFiniteScalarQuantizer(cfg.fsq_levels, cfg.num_quantizers)
                latent_dim = cfg.num_quantizers * len(cfg.fsq_levels)
            case FsqComposition.RESIDUAL:
                self.quantizer = ResidualFiniteScalarQuantizer(cfg.fsq_levels, cfg.num_quantizers, cfg.quant_dropout)
                latent_dim = len(cfg.fsq_levels)
            case unknown:
                assert_never(unknown)

        self.pre_q = nn.Linear(cfg.width, latent_dim)
        self.post_q = nn.Linear(latent_dim, cfg.width)

    @property
    def codebook_size(self) -> int:
        return self.quantizer.quantizers[0].codebook_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, indices = self.quantizer(self.pre_q(self.encoder(x)))
        return reject_dropped_codes(indices)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_q(self.quantizer.indices_to_codes(indices)))

    def forward(self, x: torch.Tensor) -> TokenizerForwardOutput:
        quantized, indices = self.quantizer(self.pre_q(self.encoder(x)))
        recon = self.decoder(self.post_q(quantized))
        return TokenizerForwardOutput(recon=recon, indices=indices, commit=x.new_zeros(()))


def feature_velocity_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    feat = F.l1_loss(recon, target)
    vel = F.l1_loss(recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1])

    return feat + vel


class RvqBaselineTokenizer(MotionTokenizerNetwork):
    def __init__(self, cfg: RvqConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = MotionFeatureEncoder(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)
        self.decoder = MotionFeatureDecoder(cfg.in_dim, cfg.width, cfg.downsample, cfg.n_resblocks)
        self.pre_q = nn.Linear(cfg.width, cfg.code_dim)
        self.post_q = nn.Linear(cfg.code_dim, cfg.width)
        self.rvq = ResidualVectorQuantizer(cfg)

    @property
    def codebook_size(self) -> int:
        return self.cfg.codebook_size

    @property
    def quantizer(self) -> ResidualVectorQuantizer:
        return self.rvq

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_q(self.encoder(x))
        _, indices, _ = self.rvq(z)
        return reject_dropped_codes(indices)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.rvq.indices_to_codes(indices)
        return self.decoder(self.post_q(codes))

    def forward(self, x: torch.Tensor) -> TokenizerForwardOutput:
        z = self.pre_q(self.encoder(x))
        quantized, indices, commit = self.rvq(z)
        recon = self.decoder(self.post_q(quantized))
        return TokenizerForwardOutput(recon=recon, indices=indices, commit=commit)


def build_tokenizer_network(
    fsq: TokenizerConfig, rvq: RvqConfig | None = None
) -> MotionTokenizerNetwork:
    match TokenizerKind(fsq.kind):
        case TokenizerKind.FSQ:
            return FsqTokenizer(fsq)
        case TokenizerKind.RVQ:
            return RvqBaselineTokenizer(rvq or RvqConfig())
        case unknown:
            assert_never(unknown)


class MotionTokenizer:
    def __init__(
        self,
        tokenizer_model: MotionTokenizerNetwork,
        downsample: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.tokenizer_model = tokenizer_model.to(device).eval()
        self.downsample = downsample
        self.device = device

    @property
    def codebook_size(self) -> int:
        return int(self.tokenizer_model.codebook_size)

    @property
    def num_codebooks(self) -> int:
        return len(self.tokenizer_model.latent_quantizer().quantizers)

    def aligned_frame_count(self, frame_count: int) -> int:
        return (frame_count // self.downsample) * self.downsample

    @torch.no_grad()
    def encode(self, motion: MotionClip) -> MotionTokens:
        usable = self.aligned_frame_count(motion.frame_count)
        if usable == 0:
            raise ValueError(
                f"clip has {motion.frame_count} frames, fewer than one token at downsample "
                f"{self.downsample}"
            )
        features = motion.features[:usable].unsqueeze(0).to(self.device)
        indices = self.tokenizer_model.encode(features)[0]
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
        features = self.tokenizer_model.decode(indices.to(self.device))[0]
        return MotionClip(features=features, frame_count=int(features.shape[0]))
