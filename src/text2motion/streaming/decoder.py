from __future__ import annotations

import queue as queue_mod
from collections import deque
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from text2motion.generation.model import MotionGeneratorModule
from text2motion.motion.representation import DIM
from text2motion.tokenization.model import TokenizerModule

STREAM_END = None


@torch.no_grad()
def measure_decoder_context(
    tokenizer: TokenizerModule, downsample: int, probe_len: int = 32, trials: int = 4
) -> tuple[int, int]:
    num_quantizers = len(tokenizer.quantizer.units)
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


def _put_drop_oldest(out_queue: "queue_mod.Queue", item: object) -> None:
    try:
        out_queue.put_nowait(item)

    except queue_mod.Full:
        try:
            out_queue.get_nowait()

        except queue_mod.Empty:
            pass

        out_queue.put_nowait(item)


class StreamingMotionDecoder:
    def __init__(
        self,
        tokenizer: TokenizerModule,
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
    def _emit(self, history: deque, pending: list[torch.Tensor], emit_tokens: int) -> torch.Tensor:
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
                yield self._emit(history, pending, self.chunk_tokens)

        while pending:
            yield self._emit(history, pending, min(self.chunk_tokens, len(pending)))

    @torch.no_grad()
    def stream(
        self,
        generator: MotionGeneratorModule,
        text_emb: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        cfg_scale: float = 1.0,
        stop_at_end: bool = False,
    ) -> Iterator[torch.Tensor]:
        yield from self.stream_tokens(
            generator.stream(
                text_emb,
                num_steps,
                temperature=temperature,
                top_p=top_p,
                cfg_scale=cfg_scale,
                stop_at_end=stop_at_end,
            )
        )


def run_producer(
    decoder: StreamingMotionDecoder,
    generator: MotionGeneratorModule,
    text_emb: torch.Tensor,
    num_steps: int,
    out_queue: "queue_mod.Queue",
    temperature: float = 1.0,
    top_p: float = 0.9,
    cfg_scale: float = 1.0,
    stop_at_end: bool = False,
) -> None:
    for chunk in decoder.stream(
        generator,
        text_emb,
        num_steps,
        temperature=temperature,
        top_p=top_p,
        cfg_scale=cfg_scale,
        stop_at_end=stop_at_end,
    ):
        _put_drop_oldest(out_queue, chunk)

    _put_drop_oldest(out_queue, STREAM_END)


def collect_stream(out_queue: "queue_mod.Queue") -> np.ndarray:
    chunks: list[np.ndarray] = []

    while True:
        item = out_queue.get()

        if item is STREAM_END:
            break

        chunks.append(item.squeeze(0).cpu().numpy())

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, DIM), dtype=np.float32)
