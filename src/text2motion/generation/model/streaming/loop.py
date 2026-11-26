from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

from text2motion.generation.contracts import SamplingConfig, StreamAnchors
from text2motion.generation.model.streaming.budget import check_anchor_request
from text2motion.generation.model.streaming.decoding import (
    project_text,
    reached_end,
    sample_step,
    start_decoding,
)

if TYPE_CHECKING:
    from text2motion.generation.model.generator import MotionTokenGenerator


class ReanchorWindow:
    def __init__(self, anchors: StreamAnchors) -> None:
        self.every = anchors.every
        self.capacity = anchors.context if anchors.every else 0
        self.recent: deque[torch.Tensor] = deque(maxlen=self.capacity)
        self.since = 0

    def record(self, feed: torch.Tensor) -> bool:
        if not self.every:
            return False
        self.recent.append(feed)
        self.since += 1
        if self.since < self.every:
            return False
        self.since = 0
        return True

    def restart(
        self,
        model: MotionTokenGenerator,
        index: int,
        anchors: StreamAnchors,
        text_emb: torch.Tensor,
        guided: bool,
        batch: int,
    ):
        text = text_emb if anchors.text_at is None else anchors.text_at(index)
        prefix, text_vector = project_text(model, text, guided)
        context = model.reset_context(self.recent, index, anchors.context, anchors.context_at)
        return start_decoding(model, context, prefix, text_vector, batch, text_emb.device)


def stream_indices(
    model: MotionTokenGenerator,
    text_emb: torch.Tensor,
    steps: int,
    sampling: SamplingConfig,
    uniforms: torch.Tensor | None,
    anchors: StreamAnchors,
):
    guided = sampling.cfg_scale != 1.0
    check_anchor_request(anchors)
    prefix, text_vector = project_text(model, text_emb, guided)
    capacity = anchors.context if anchors.every else 0
    model.check_stream_budget(steps, prefix.size(1), uniforms, anchors.every, capacity)
    window = ReanchorWindow(anchors)
    batch = prefix.size(0)
    logits, decode_step = start_decoding(model, [], prefix, text_vector, batch, text_emb.device)
    index = 0
    while steps == 0 or index < steps:
        tokens = sample_step(model, logits, sampling, uniforms, index)
        if reached_end(model.end_id, sampling, tokens):
            return
        yield tokens
        index += 1
        if steps != 0 and index >= steps:
            return
        feed = torch.cat([tokens, tokens], dim=0) if guided else tokens
        if not window.record(feed):
            logits = decode_step(feed)
            continue
        logits, decode_step = window.restart(model, index, anchors, text_emb, guided, batch)
