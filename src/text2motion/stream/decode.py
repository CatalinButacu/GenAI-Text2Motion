"""Streaming decode: bounded-state generator -> token windows -> motion frames -> a viewer queue.

The generator (Contribution B) streams tokens with O(1) state. Here we decode them to 263 motion
frames in fixed token-windows AS THEY ARRIVE (chunk-by-chunk, never waiting for the full sequence)
and push them onto a bounded queue for the studio. Each window is decoded independently, so the
streamed frames are deterministic and concatenate exactly to the per-window decode. (A non-causal
conv decoder has mild window-boundary artifacts vs a single full decode; overlap-add is a future
refinement — the bounded-memory CLAIM is about the generator, not this lightweight decode.)
See `.claude/skills/streaming-decode`.
"""

import queue as queue_mod
from collections.abc import Iterable, Iterator

import torch

from text2motion.model.generator import MotionGenerator
from text2motion.model.tokenizer import ResidualFsqTokenizer

STREAM_END = None  # sentinel pushed to the queue when generation finishes


def _put_drop_oldest(out_queue: "queue_mod.Queue", item: object) -> None:
    """Non-blocking put with drop-to-latest: if the queue is full, discard the oldest item to make
    room. The producer NEVER stalls (the bounded-memory contract) -- a lagging consumer loses old
    frames, not the producer's pace. Used for both frame chunks and the STREAM_END sentinel."""
    try:
        out_queue.put_nowait(item)

    except queue_mod.Full:
        try:
            out_queue.get_nowait()

        except queue_mod.Empty:
            pass

        out_queue.put_nowait(item)


class StreamingMotionDecoder:
    """Turns a stream of per-step tokens into a stream of denormalised 263 motion-frame chunks."""

    def __init__(
        self,
        tokenizer: ResidualFsqTokenizer,
        mean: torch.Tensor,
        std: torch.Tensor,
        chunk_tokens: int = 4,
    ) -> None:
        self.tokenizer = tokenizer.eval()
        self.mean = mean  # (263,)
        self.std = std  # (263,)
        self.chunk_tokens = chunk_tokens

    @torch.no_grad()
    def decode_tokens(self, token_seq: torch.Tensor) -> torch.Tensor:
        """token_seq (B, N, R) -> denormalised motion (B, N*downsample, 263)."""
        return self.tokenizer.decode(token_seq) * self.std + self.mean

    @torch.no_grad()
    def stream_tokens(self, token_iter: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        """Consume per-step tokens (B, R); yield a (B, chunk_tokens*downsample, 263) frame chunk
        each time a window fills, plus a final partial window."""
        window: list[torch.Tensor] = []

        for tokens in token_iter:
            window.append(tokens)

            if len(window) == self.chunk_tokens:
                yield self.decode_tokens(torch.stack(window, dim=1))
                window = []

        if window:
            yield self.decode_tokens(torch.stack(window, dim=1))

    @torch.no_grad()
    def stream(
        self,
        generator: MotionGenerator,
        text_emb: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Iterator[torch.Tensor]:
        """End-to-end: bounded-state token generation -> frame chunks."""
        yield from self.stream_tokens(generator.stream(text_emb, num_steps, temperature, top_p))


def run_producer(
    decoder: StreamingMotionDecoder,
    generator: MotionGenerator,
    text_emb: torch.Tensor,
    num_steps: int,
    out_queue: "queue_mod.Queue",
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> None:
    """Background producer: push frame chunks onto a bounded queue (drop-to-latest if the consumer
    lags — never stall generation), then push STREAM_END. Run in a daemon thread."""
    for chunk in decoder.stream(generator, text_emb, num_steps, temperature, top_p):
        _put_drop_oldest(out_queue, chunk)

    _put_drop_oldest(out_queue, STREAM_END)  # sentinel must never block either
