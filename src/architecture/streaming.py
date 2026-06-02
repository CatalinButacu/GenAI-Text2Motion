"""Streaming inference primitives for TextToMotionSSM.

State container + carry_over factory. Per-step advance lives on
TextToMotionSSM.stream_step. Requires bidirectional=False (BiMambaLayer
has no causal step path).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StreamingState:
    """Per-action streaming state for one batch element (or batch of size B)."""

    cond: torch.Tensor  # (B, d_model) text condition; reused per step until carry_over
    layer_h: list[torch.Tensor]  # len n_layers; each (B, d_inner, d_state) SSM hidden state
    layer_conv: list[torch.Tensor | None]  # len n_layers; (B, d_inner, d_conv-1) or None at t=0
    max_steps: int  # = model.latent_length; caller stops emitting at this point
    latent_step: int = 0  # count emitted in current action; resets on carry_over

    def carry_over(self, new_cond: torch.Tensor) -> StreamingState:
        """Continue with a new action: replace cond, preserve hidden state, reset latent_step."""
        self.cond = new_cond
        self.latent_step = 0

        return self
