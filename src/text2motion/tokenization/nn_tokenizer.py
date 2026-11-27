from __future__ import annotations

from typing import assert_never

import torch
from torch import nn

from text2motion.motion.contracts import MotionClip, MotionTokens
from text2motion.tokenization.contracts import (
    FiniteScalarStackLayout,
    ResidualVectorQuantizationConfig,
    TokenizerConfig,
    TokenizerKind,
    TokenizerReconstruction,
)
from text2motion.tokenization.fsq_quantizer import (
    GroupedFiniteScalarQuantizer,
    ResidualFiniteScalarQuantizer,
)
from text2motion.tokenization.network.base import TokenizerNetwork
from text2motion.tokenization.network.losses import (
    feature_velocity_reconstruction_loss,
    heading_reconstruction_loss,
    masked_mean_abs,
)
from text2motion.tokenization.network.motion_convolution import (
    CausalConvolution1d,
    FrameSequenceConvolution,
    MotionFeatureDecoder,
    MotionFeatureEncoder,
    TemporalResidualBlock,
    downsample_stage_count,
    frame_convolution,
)
from text2motion.tokenization.quantizer_base import QuantizerStack
from text2motion.tokenization.vector_quantizer import ResidualVectorQuantizer

__all__ = [
    "CausalConvolution1d",
    "FrameSequenceConvolution",
    "MotionFeatureDecoder",
    "MotionFeatureEncoder",
    "TemporalResidualBlock",
    "TokenizerNetwork",
    "downsample_stage_count",
    "feature_velocity_reconstruction_loss",
    "frame_convolution",
    "heading_reconstruction_loss",
    "masked_mean_abs",
]


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
