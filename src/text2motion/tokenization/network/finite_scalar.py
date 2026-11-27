from __future__ import annotations

from typing import assert_never

import torch
from torch import nn

from text2motion.tokenization.contracts import (
    FiniteScalarStackLayout,
    TokenizerConfig,
    TokenizerReconstruction,
)
from text2motion.tokenization.fsq_quantizer import (
    GroupedFiniteScalarQuantizer,
    ResidualFiniteScalarQuantizer,
)
from text2motion.tokenization.network.base import TokenizerNetwork
from text2motion.tokenization.network.motion_convolution import (
    MotionFeatureDecoder,
    MotionFeatureEncoder,
)
from text2motion.tokenization.quantizer_base import QuantizerStack


def finite_scalar_quantizer(config: TokenizerConfig) -> tuple[QuantizerStack, int]:
    levels = config.fsq_levels
    match FiniteScalarStackLayout(config.quantizer):
        case FiniteScalarStackLayout.GROUPED:
            quantizer = GroupedFiniteScalarQuantizer(levels, config.num_quantizers)
            return quantizer, config.num_quantizers * len(levels)
        case FiniteScalarStackLayout.RESIDUAL:
            quantizer = ResidualFiniteScalarQuantizer(
                levels, config.num_quantizers, config.quant_dropout
            )
            return quantizer, len(levels)
        case unknown:
            assert_never(unknown)


def zero_heading_head(width: int) -> nn.Linear:
    head = nn.Linear(width, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


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
        self.quantizer, latent_width = finite_scalar_quantizer(config)
        self.pre_q = nn.Linear(config.width, latent_width)
        self.post_q = nn.Linear(latent_width, config.width)
        self.heading_weight = config.heading_weight if config.heading_correction else 0.0
        self.heading_head = None
        if config.heading_correction:
            self.heading_head = zero_heading_head(config.width)

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
