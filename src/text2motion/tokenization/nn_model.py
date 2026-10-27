from __future__ import annotations

from typing import assert_never

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.motion.model import MotionClip, MotionTokens
from text2motion.tokenization.contracts import (
    FiniteScalarStackLayout,
    ResidualVectorQuantizationConfig,
    TokenizerConfig,
    TokenizerKind,
    TokenizerReconstruction,
)
from text2motion.tokenization.nn_quantizers import (
    GroupedFiniteScalarQuantizer,
    QuantizerStack,
    ResidualFiniteScalarQuantizer,
    ResidualVectorQuantizer,
    reject_dropped_codes,
)


class TokenizerNetwork(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    pre_q: nn.Module
    post_q: nn.Module

    def quantizer_stack(self) -> QuantizerStack:
        quantizer = self.quantizer
        if not isinstance(quantizer, QuantizerStack):
            raise TypeError("the tokenizer network has no quantizer stack")
        return quantizer

    @property
    def codebook_size(self) -> int:
        return self.quantizer_stack().quantizers[0].codebook_size

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        _, indices, _ = self.quantizer_stack()(self.pre_q(self.encoder(motion)))
        return reject_dropped_codes(indices)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        code_vectors = self.quantizer_stack().indices_to_codes(indices)
        return self.decoder(self.post_q(code_vectors))

    def forward(self, motion: torch.Tensor) -> TokenizerReconstruction:
        latent = self.pre_q(self.encoder(motion))
        quantized, indices, commitment_loss = self.quantizer_stack()(latent)
        return TokenizerReconstruction(
            reconstruction=self.decoder(self.post_q(quantized)),
            token_indices=indices,
            commitment_loss=commitment_loss,
        )


def downsample_stage_count(downsample: int) -> int:
    if downsample < 1 or downsample & (downsample - 1):
        raise ValueError(
            f"tokenizer downsample must be a power of two, got {downsample}; the encoder builds "
            f"stride-2 stages and any other value would silently change the frame rate per token"
        )
    return downsample.bit_length() - 1


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MotionFeatureEncoder(nn.Module):
    def __init__(
        self, feature_count: int, width: int, downsample: int, residual_block_count: int
    ) -> None:
        super().__init__()
        downsample_stages = downsample_stage_count(downsample)
        layers: list[nn.Module] = [nn.Conv1d(feature_count, width, 3, padding=1), nn.ReLU()]

        for _ in range(downsample_stages):
            layers += [nn.Conv1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        for _ in range(residual_block_count):
            layers.append(TemporalResidualBlock(width))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class MotionFeatureDecoder(nn.Module):
    def __init__(
        self, feature_count: int, width: int, downsample: int, residual_block_count: int
    ) -> None:
        super().__init__()
        upsample_stages = downsample_stage_count(downsample)
        layers: list[nn.Module] = []

        for _ in range(residual_block_count):
            layers.append(TemporalResidualBlock(width))

        for _ in range(upsample_stages):
            layers += [nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        layers.append(nn.Conv1d(width, feature_count, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class FiniteScalarQuantizationTokenizer(TokenizerNetwork):
    quantizer: QuantizerStack

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        self.encoder = MotionFeatureEncoder(
            config.in_dim, config.width, config.downsample, config.n_resblocks
        )
        self.decoder = MotionFeatureDecoder(
            config.in_dim, config.width, config.downsample, config.n_resblocks
        )

        levels = config.fsq_levels
        match FiniteScalarStackLayout(config.quantizer):
            case FiniteScalarStackLayout.GROUPED:
                self.quantizer = GroupedFiniteScalarQuantizer(levels, config.num_quantizers)
                latent_width = config.num_quantizers * len(levels)
            case FiniteScalarStackLayout.RESIDUAL:
                self.quantizer = ResidualFiniteScalarQuantizer(
                    levels, config.num_quantizers, config.quant_dropout
                )
                latent_width = len(levels)
            case unknown:
                assert_never(unknown)

        self.pre_q = nn.Linear(config.width, latent_width)
        self.post_q = nn.Linear(latent_width, config.width)


def feature_velocity_reconstruction_loss(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    feature_loss = F.l1_loss(reconstruction, target)
    reconstruction_velocity = reconstruction[:, 1:] - reconstruction[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity_loss = F.l1_loss(reconstruction_velocity, target_velocity)

    return feature_loss + velocity_loss


class ResidualVectorQuantizationTokenizer(TokenizerNetwork):
    def __init__(self, config: ResidualVectorQuantizationConfig) -> None:
        super().__init__()
        self.encoder = MotionFeatureEncoder(
            config.in_dim, config.width, config.downsample, config.n_resblocks
        )
        self.decoder = MotionFeatureDecoder(
            config.in_dim, config.width, config.downsample, config.n_resblocks
        )
        self.pre_q = nn.Linear(config.width, config.code_dim)
        self.post_q = nn.Linear(config.code_dim, config.width)
        self.rvq = ResidualVectorQuantizer(config)

    @property
    def quantizer(self) -> ResidualVectorQuantizer:
        return self.rvq


def build_tokenizer_network(
    finite_scalar_config: TokenizerConfig,
    residual_vector_config: ResidualVectorQuantizationConfig | None = None,
) -> TokenizerNetwork:
    match TokenizerKind(finite_scalar_config.kind):
        case TokenizerKind.FINITE_SCALAR:
            return FiniteScalarQuantizationTokenizer(finite_scalar_config)
        case TokenizerKind.RESIDUAL_VECTOR:
            residual_config = residual_vector_config or ResidualVectorQuantizationConfig()
            return ResidualVectorQuantizationTokenizer(residual_config)
        case unknown:
            assert_never(unknown)


class FrozenMotionTokenizer:
    def __init__(
        self,
        network: TokenizerNetwork,
        downsample: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.network = network.to(device).eval()
        self.downsample = downsample
        self.device = device

    @property
    def codebook_size(self) -> int:
        return int(self.network.codebook_size)

    @property
    def num_codebooks(self) -> int:
        return len(self.network.quantizer_stack().quantizers)

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
        indices = self.network.encode(features)[0]
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
        features = self.network.decode(indices.to(self.device))[0]
        return MotionClip(features=features, frame_count=int(features.shape[0]))
