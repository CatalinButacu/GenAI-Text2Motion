from __future__ import annotations

import torch
from torch import nn

from text2motion.tokenization.contracts import TokenizerReconstruction
from text2motion.tokenization.quantizer_base import QuantizerStack, reject_dropped_codes


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
