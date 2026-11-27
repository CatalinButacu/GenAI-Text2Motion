from __future__ import annotations

import torch

from text2motion.motion.contracts import MotionClip, MotionTokens
from text2motion.tokenization.network.base import TokenizerNetwork


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
