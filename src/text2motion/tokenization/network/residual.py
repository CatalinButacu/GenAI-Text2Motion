from __future__ import annotations

from torch import nn

from text2motion.tokenization.contracts import ResidualVectorQuantizationConfig
from text2motion.tokenization.network.base import TokenizerNetwork
from text2motion.tokenization.network.motion_convolution import (
    MotionFeatureDecoder,
    MotionFeatureEncoder,
)
from text2motion.tokenization.vector_quantizer import ResidualVectorQuantizer


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
