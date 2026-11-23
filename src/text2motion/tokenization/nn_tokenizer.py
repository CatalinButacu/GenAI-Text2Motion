from __future__ import annotations

from typing import assert_never

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.motion.contracts import MotionClip, MotionTokens
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


############################################################
#       TokenizerNetwork
############################################################
class TokenizerNetwork(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    pre_q: nn.Module
    post_q: nn.Module
    downsample: int
    heading_weight: float = 0.0
    heading_head: nn.Linear | None

    def checked_frame_count(self, motion: torch.Tensor) -> torch.Tensor:
        def reject_non_finite_motion(motion: torch.Tensor) -> torch.Tensor:
            if not bool(torch.isfinite(motion).all()):
                raise ValueError("the motion holds NaN or Inf values; fix the clip before encode")
            return motion

        frames = motion.shape[1]
        if frames < self.downsample or frames % self.downsample:
            raise ValueError(
                f"the motion has {frames} frames; the tokenizer needs a positive multiple of "
                f"downsample {self.downsample}, so cut the clip before encode"
            )
        return reject_non_finite_motion(motion)

    def quantizer_stack(self) -> QuantizerStack:
        quantizer = self.quantizer
        if not isinstance(quantizer, QuantizerStack):
            raise TypeError("the tokenizer network has no quantizer stack")
        return quantizer

    @property
    def codebook_size(self) -> int:
        return self.quantizer_stack().quantizers[0].codebook_size

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        checked = self.checked_frame_count(motion)
        _, indices, _ = self.quantizer_stack()(self.pre_q(self.encoder(checked)))
        if bool((indices < 0).any()):
            reject_dropped_codes(indices)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        code_vectors = self.quantizer_stack().indices_to_codes(indices)
        return self.decoder(self.post_q(code_vectors))

    def forward(self, motion: torch.Tensor) -> TokenizerReconstruction:
        latent = self.pre_q(self.encoder(self.checked_frame_count(motion)))
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


############################################################
#       TemporalResidualBlock
############################################################
class CausalConvolution1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


def frame_convolution(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, causal: bool
) -> nn.Module:
    if causal:
        return CausalConvolution1d(in_channels, out_channels, kernel_size, stride)
    return nn.Conv1d(
        in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2
    )


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, causal: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            frame_convolution(channels, channels, 3, 1, causal),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


############################################################
#       FrameSequenceConvolution
############################################################
class FrameSequenceConvolution(nn.Module):
    net: nn.Sequential

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames.transpose(1, 2)).transpose(1, 2)


############################################################
#       MotionFeatureEncoder
############################################################
class MotionFeatureEncoder(FrameSequenceConvolution):
    def __init__(
        self,
        feature_count: int,
        width: int,
        downsample: int,
        residual_block_count: int,
        causal: bool = False,
    ) -> None:
        super().__init__()
        downsample_stages = downsample_stage_count(downsample)
        layers: list[nn.Module] = [
            frame_convolution(feature_count, width, 3, 1, causal),
            nn.ReLU(),
        ]

        for _ in range(downsample_stages):
            layers += [frame_convolution(width, width, 4, 2, causal), nn.ReLU()]

        for _ in range(residual_block_count):
            layers.append(TemporalResidualBlock(width, causal))

        self.net = nn.Sequential(*layers)


############################################################
#       MotionFeatureDecoder
############################################################
class MotionFeatureDecoder(FrameSequenceConvolution):
    def __init__(
        self,
        feature_count: int,
        width: int,
        downsample: int,
        residual_block_count: int,
        causal: bool = False,
    ) -> None:
        super().__init__()
        upsample_stages = downsample_stage_count(downsample)
        layers: list[nn.Module] = []

        for _ in range(residual_block_count):
            layers.append(TemporalResidualBlock(width, causal))

        for _ in range(upsample_stages):
            if causal:
                layers += [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    frame_convolution(width, width, 3, 1, causal),
                    nn.ReLU(),
                ]
            else:
                layers += [nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        layers.append(frame_convolution(width, feature_count, 3, 1, causal))
        self.net = nn.Sequential(*layers)


############################################################
#       FiniteScalarQuantizationTokenizer
############################################################
class FiniteScalarQuantizationTokenizer(TokenizerNetwork):
    quantizer: QuantizerStack

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        self.downsample = config.downsample
        self.encoder = MotionFeatureEncoder(
            config.in_dim, config.width, config.downsample, config.n_resblocks, config.causal
        )
        self.decoder = MotionFeatureDecoder(
            config.in_dim, config.width, config.downsample, config.n_resblocks, config.causal
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
        self.heading_weight = config.heading_weight if config.heading_correction else 0.0
        self.heading_head = None
        if config.heading_correction:
            head = nn.Linear(config.width, 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.heading_head = head

    def correct_heading(self, reconstruction: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if self.heading_head is None:
            return reconstruction
        correction = self.heading_head(latent).repeat_interleave(self.downsample, dim=1)
        turning = reconstruction[..., :1] + correction
        return torch.cat([turning, reconstruction[..., 1:]], dim=-1)

    def forward(self, motion: torch.Tensor) -> TokenizerReconstruction:
        latent = self.pre_q(self.encoder(self.checked_frame_count(motion)))
        quantized, indices, commitment_loss = self.quantizer_stack()(latent)
        decoded_latent = self.post_q(quantized)
        reconstruction = self.decoder(decoded_latent)
        return TokenizerReconstruction(
            reconstruction=self.correct_heading(reconstruction, decoded_latent),
            token_indices=indices,
            commitment_loss=commitment_loss,
        )


def heading_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    frames = target.shape[1]
    if lengths is None:
        mask = torch.ones(target.shape[:2], device=target.device, dtype=target.dtype)
    else:
        mask = (
            torch.arange(frames, device=target.device)[None, :] < lengths.to(target.device)[:, None]
        ).to(target.dtype)
    turning_error = (reconstruction[..., 0] - target[..., 0]) * mask
    accumulated = torch.cumsum(turning_error, dim=1)
    return (accumulated.abs() * mask).sum() / mask.sum().clamp(min=1.0)


def feature_velocity_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if reconstruction.shape != target.shape or target.shape[0] == 0 or target.shape[1] < 2:
        raise ValueError(
            f"the reconstruction loss needs equal shapes with at least 2 frames, got "
            f"{tuple(reconstruction.shape)} and {tuple(target.shape)}"
        )
    if lengths is None:
        feature_loss = F.l1_loss(reconstruction, target)
        reconstruction_velocity = reconstruction[:, 1:] - reconstruction[:, :-1]
        target_velocity = target[:, 1:] - target[:, :-1]
        return feature_loss + F.l1_loss(reconstruction_velocity, target_velocity)
    frame_mask = (
        torch.arange(target.shape[1], device=target.device)[None, :]
        < lengths.to(target.device)[:, None]
    ).to(target.dtype)
    feature_loss = masked_mean_abs(reconstruction - target, frame_mask)
    reconstruction_velocity = reconstruction[:, 1:] - reconstruction[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity_mask = frame_mask[:, 1:] * frame_mask[:, :-1]
    velocity_loss = masked_mean_abs(reconstruction_velocity - target_velocity, velocity_mask)
    return feature_loss + velocity_loss


def masked_mean_abs(difference: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    per_frame = difference.abs().mean(-1)
    return (per_frame * frame_mask).sum() / frame_mask.sum().clamp(min=1.0)


############################################################
#       ResidualVectorQuantizationTokenizer
############################################################
class ResidualVectorQuantizationTokenizer(TokenizerNetwork):
    def __init__(self, config: ResidualVectorQuantizationConfig) -> None:
        super().__init__()
        self.downsample = config.downsample
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
            if residual_vector_config is None:
                raise ValueError(
                    "an RVQ tokenizer needs its ResidualVectorQuantizationConfig; pass rvq_baseline"
                )
            return ResidualVectorQuantizationTokenizer(residual_vector_config)
        case unknown:
            assert_never(unknown)


############################################################
#       FrozenMotionTokenizer
############################################################
class FrozenMotionTokenizer:
    def __init__(
        self,
        network: TokenizerNetwork,
        device: str | torch.device = "cpu",
    ) -> None:
        self.network = network.to(device).eval()
        self.downsample = network.downsample
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
        if motion.features.dim() != 2 or motion.features.shape[0] < motion.frame_count:
            raise ValueError(
                f"the clip says {motion.frame_count} frames, but its features have shape "
                f"{tuple(motion.features.shape)}"
            )
        features = motion.features[:usable].unsqueeze(0).to(self.device)
        self.network.eval()
        indices = self.network.encode(features)[0]
        return MotionTokens(
            indices=indices,
            frame_count=usable,
        )

    @torch.no_grad()
    def decode(self, tokens: MotionTokens) -> MotionClip:
        indices = tokens.indices
        if indices.dim() not in (2, 3):
            raise ValueError(
                f"token indices must have shape [steps, codebooks] or [batch, steps, codebooks], "
                f"got {tuple(indices.shape)}"
            )
        if indices.dim() == 3 and indices.shape[0] != 1:
            raise ValueError(
                f"decode returns one clip; got a batch of {indices.shape[0]} token sequences"
            )
        if indices.dim() == 2:
            indices = indices.unsqueeze(0)
        if tokens.frame_count != indices.shape[1] * self.downsample:
            raise ValueError(
                f"{indices.shape[1]} token steps give {indices.shape[1] * self.downsample} frames, "
                f"but the tokens say {tokens.frame_count}"
            )
        self.network.eval()
        features = self.network.decode(indices.to(self.device))[0]
        return MotionClip(features=features, frame_count=int(features.shape[0]))
