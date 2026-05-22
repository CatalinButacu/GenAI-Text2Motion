"""Streaming inference primitives for TextToMotionSSM.

The offline forward pass scans every latent step in parallel (50 steps for a
200-frame clip). Streaming inference instead emits one latent step at a time
and carries the recurrent SSM hidden state across calls, so:

  - latency-to-first-frame is O(1) wrt clip length
  - per-step compute is O(d_model * d_state), independent of the number of
    latent steps already emitted
  - hidden state can be reused across action transitions, giving the
    "carry-over continuation" thesis claim

This module is intentionally minimal: a state container and a single
:func:`stream_begin` factory. The per-step advance lives on
:class:`TextToMotionSSM.stream_step` so it can access the layers, the FiLM
blocks and the decoder directly.

Requires the model to be built with ``bidirectional=False`` because
:class:`BiMambaLayer` has no causal ``step`` path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class StreamingState:
    """Per-action streaming state for one batch element (or batch of size B).

    Attributes:
        cond: ``(B, d_model)`` text condition vector, computed once at
            :func:`stream_begin` time and reused for every step until the
            action changes.
        layer_h: ``list[torch.Tensor]`` of length ``n_layers``; each entry is
            ``(B, d_inner, d_state)`` — the recurrent SSM hidden state for one
            Mamba layer.
        layer_conv: ``list[torch.Tensor | None]`` of length ``n_layers``; each
            entry is the rolling depthwise-conv buffer ``(B, d_inner, d_conv-1)``
            or ``None`` on the first step.
        latent_step: integer count of latent steps already emitted in the
            current action. Resets to 0 in :func:`stream_begin`.
        max_steps: ``self.latent_length`` of the model — caller is expected to
            stop emitting at this point. Stored on the state so the caller
            doesn't have to thread the model in.
    """

    cond: torch.Tensor
    layer_h: list[torch.Tensor]
    layer_conv: list[torch.Tensor | None] = field(default_factory=list)
    latent_step: int = 0
    max_steps: int = 0

    def carry_over(self, new_cond: torch.Tensor) -> StreamingState:
        """Begin a new action, keeping the SSM hidden state from the previous one.

        Updates ``cond`` to the new action's text condition but preserves
        ``layer_h`` and ``layer_conv`` so the avatar continues smoothly. The
        ``latent_step`` counter resets so positional embeddings restart at 0
        for the new action.
        """
        self.cond = new_cond
        self.latent_step = 0

        return self
