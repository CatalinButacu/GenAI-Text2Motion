from __future__ import annotations

from collections.abc import Iterator
from enum import Enum

import torch

from text2motion.generation.contracts import DEFAULT_SAMPLING, SamplingConfig, StreamAnchors
from text2motion.generation.decoder import StreamingMotionDecoder
from text2motion.generation.model.generator import MotionTokenGenerator


############################################################
#       Caption steering
############################################################
class SteeringMode(Enum):
    REANCHOR = "reanchor"
    CROSSFADE = "crossfade"


def require_switch_inside(switch_token: int, num_steps: int) -> None:
    if not 0 < switch_token < num_steps:
        raise ValueError("switch_token must lie inside the stream")


def reanchor_stream(
    generator: MotionTokenGenerator,
    decoder: StreamingMotionDecoder,
    old_text: torch.Tensor,
    new_text: torch.Tensor,
    num_steps: int,
    switch_token: int,
    sampling: SamplingConfig,
    anchor_every: int,
    anchor_context: int,
) -> Iterator[tuple[int, torch.Tensor]]:
    if switch_token % anchor_every:
        raise ValueError("switch_token must be a multiple of anchor_every")
    anchors = StreamAnchors(
        every=anchor_every,
        context=anchor_context,
        text_at=lambda index: old_text if index < switch_token else new_text,
    )
    frame_index = 0
    for chunk in decoder.stream_tokens(
        generator.stream_token_indices(old_text, num_steps, sampling, anchors=anchors)
    ):
        yield frame_index, chunk
        frame_index += chunk.shape[1]


def blend_weight(
    chunk: torch.Tensor, frame_index: int, start: int, blend_frames: int
) -> torch.Tensor:
    frames = torch.arange(frame_index, frame_index + chunk.shape[1], device=chunk.device)
    return ((frames - start + 1) / blend_frames).clamp(0.0, 1.0)[None, :, None]


def crossfade_stream(
    generator: MotionTokenGenerator,
    decoder: StreamingMotionDecoder,
    old_text: torch.Tensor,
    new_text: torch.Tensor,
    num_steps: int,
    switch_token: int,
    sampling: SamplingConfig,
    blend_frames: int,
) -> Iterator[tuple[int, torch.Tensor]]:
    if blend_frames < 1:
        raise ValueError("blend_frames must be positive")
    start = switch_token * decoder.downsample
    old_chunks = decoder.stream_tokens(
        generator.stream_token_indices(old_text, num_steps, sampling)
    )
    new_chunks = decoder.stream_tokens(
        generator.stream_token_indices(new_text, num_steps, sampling)
    )
    frame_index = 0
    for old_chunk, new_chunk in zip(old_chunks, new_chunks, strict=True):
        weight = blend_weight(old_chunk, frame_index, start, blend_frames)
        yield frame_index, old_chunk * (1.0 - weight) + new_chunk * weight
        frame_index += old_chunk.shape[1]


@torch.no_grad()
def steer_stream(
    generator: MotionTokenGenerator,
    decoder: StreamingMotionDecoder,
    old_text: torch.Tensor,
    new_text: torch.Tensor,
    num_steps: int,
    switch_token: int,
    mode: SteeringMode,
    sampling: SamplingConfig = DEFAULT_SAMPLING,
    anchor_every: int = 4,
    anchor_context: int = 8,
    blend_frames: int = 16,
) -> Iterator[tuple[int, torch.Tensor]]:
    require_switch_inside(switch_token, num_steps)
    if mode is SteeringMode.REANCHOR:
        yield from reanchor_stream(
            generator,
            decoder,
            old_text,
            new_text,
            num_steps,
            switch_token,
            sampling,
            anchor_every,
            anchor_context,
        )
        return
    yield from crossfade_stream(
        generator, decoder, old_text, new_text, num_steps, switch_token, sampling, blend_frames
    )
