from __future__ import annotations

import contextlib
import queue as queue_mod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

import torch

from text2motion.generation.contracts import DEFAULT_SAMPLING, SamplingConfig, StreamAnchors
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.tokenization.nn_tokenizer import TokenizerNetwork


@torch.no_grad()
def measure_decoder_context(
    tokenizer: TokenizerNetwork, downsample: int, probe_len: int = 32, trials: int = 4
) -> tuple[int, int]:
    num_quantizers = len(tokenizer.quantizer_stack().quantizers)
    codebook_size = tokenizer.codebook_size
    device = next(tokenizer.parameters()).device
    centre = probe_len // 2
    left_tokens = 0
    right_tokens = 0

    for trial in range(trials):
        generator = torch.Generator().manual_seed(trial)
        indices = torch.randint(
            0, codebook_size, (1, probe_len, num_quantizers), generator=generator
        ).to(device)
        perturbed = indices.clone()
        perturbed[0, centre] = (perturbed[0, centre] + 1) % codebook_size
        delta = (tokenizer.decode(indices) - tokenizer.decode(perturbed)).abs().sum(-1)[0]
        changed = torch.nonzero(delta > 0).flatten()
        if changed.numel() == 0:
            continue
        first_frame = int(changed.min())
        last_frame = int(changed.max())
        left_frames = centre * downsample - first_frame
        right_frames = last_frame - ((centre + 1) * downsample - 1)
        left_tokens = max(left_tokens, -(-left_frames // downsample))
        right_tokens = max(right_tokens, -(-right_frames // downsample))

    if left_tokens == 0 and right_tokens == 0:
        raise RuntimeError(
            "decoder context probe measured a zero receptive field; the probe failed rather than "
            "the decoder being pointwise -- check tokenizer.decode is wired and the codebook is real"
        )

    return max(left_tokens, 0), max(right_tokens, 0)


def put_drop_oldest(out_queue: "queue_mod.Queue", item: object) -> None:
    try:
        out_queue.put_nowait(item)

    except queue_mod.Full:
        with contextlib.suppress(queue_mod.Empty):
            out_queue.get_nowait()

        out_queue.put_nowait(item)


############################################################
#       StreamingMotionDecoder
############################################################
class StreamingMotionDecoder:
    def __init__(
        self,
        tokenizer: TokenizerNetwork,
        downsample: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        chunk_tokens: int = 4,
        left_context: int | None = None,
        lookahead: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer.eval()
        self.downsample = downsample
        self.mean = mean
        self.std = std
        self.chunk_tokens = chunk_tokens

        if left_context is None or lookahead is None:
            measured_left, measured_right = measure_decoder_context(tokenizer, downsample)
            left_context = measured_left if left_context is None else left_context
            lookahead = measured_right if lookahead is None else lookahead
        self.left_context = left_context
        self.lookahead = lookahead

    @property
    def lookahead_frames(self) -> int:
        return self.lookahead * self.downsample

    @property
    def state_tokens(self) -> int:
        return self.left_context + self.chunk_tokens + self.lookahead

    @torch.no_grad()
    def decode_tokens(self, token_seq: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(token_seq) * self.std + self.mean

    @torch.no_grad()
    def emit(self, history: deque, pending: list[torch.Tensor], emit_tokens: int) -> torch.Tensor:
        downsample = self.downsample
        past = list(history)
        window = torch.stack(past + pending, dim=1)
        frames = self.decode_tokens(window)
        start = len(past) * downsample
        emitted = frames[:, start : start + emit_tokens * downsample]

        for _ in range(emit_tokens):
            history.append(pending.pop(0))

        return emitted

    @torch.no_grad()
    def stream_tokens(self, token_iter: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        history: deque = deque(maxlen=self.left_context)
        pending: list[torch.Tensor] = []

        for tokens in token_iter:
            pending.append(tokens)

            while len(pending) >= self.chunk_tokens + self.lookahead:
                yield self.emit(history, pending, self.chunk_tokens)

        while pending:
            yield self.emit(history, pending, min(self.chunk_tokens, len(pending)))

    @torch.no_grad()
    def stream(
        self,
        generator: MotionTokenGenerator,
        text_emb: torch.Tensor,
        num_steps: int,
        sampling: SamplingConfig = DEFAULT_SAMPLING,
    ) -> Iterator[torch.Tensor]:
        yield from self.stream_tokens(
            generator.stream_token_indices(text_emb, num_steps, sampling=sampling)
        )


############################################################
#       Caption steering
############################################################
class SteeringMode(Enum):
    REANCHOR = "reanchor"
    CROSSFADE = "crossfade"


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
    if not 0 < switch_token < num_steps:
        raise ValueError("switch_token must lie inside the stream")
    frame_index = 0
    if mode is SteeringMode.REANCHOR:
        if switch_token % anchor_every:
            raise ValueError("switch_token must be a multiple of anchor_every")
        anchors = StreamAnchors(
            every=anchor_every,
            context=anchor_context,
            text_at=lambda index: old_text if index < switch_token else new_text,
        )
        for chunk in decoder.stream_tokens(
            generator.stream_token_indices(old_text, num_steps, sampling, anchors=anchors)
        ):
            yield frame_index, chunk
            frame_index += chunk.shape[1]
        return
    if blend_frames < 1:
        raise ValueError("blend_frames must be positive")
    start = switch_token * decoder.downsample
    old_chunks = decoder.stream_tokens(
        generator.stream_token_indices(old_text, num_steps, sampling)
    )
    new_chunks = decoder.stream_tokens(
        generator.stream_token_indices(new_text, num_steps, sampling)
    )
    for old_chunk, new_chunk in zip(old_chunks, new_chunks, strict=True):
        frames = torch.arange(
            frame_index, frame_index + old_chunk.shape[1], device=old_chunk.device
        )
        weight = ((frames - start + 1) / blend_frames).clamp(0.0, 1.0)[None, :, None]
        yield frame_index, old_chunk * (1.0 - weight) + new_chunk * weight
        frame_index += old_chunk.shape[1]
