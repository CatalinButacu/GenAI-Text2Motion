import queue as queue_mod
from collections.abc import Iterable, Iterator

import numpy as np
import torch

from text2motion.model.generator import MotionGenerator
from text2motion.model.tokenizer import ResidualFsqTokenizer

STREAM_END = None  # sentinel pushed to the queue when generation finishes


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
        return self.tokenizer.decode(token_seq) * self.std + self.mean

    @torch.no_grad()
    def stream_tokens(self, token_iter: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
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
    for chunk in decoder.stream(generator, text_emb, num_steps, temperature, top_p):
        _put_drop_oldest(out_queue, chunk)

    _put_drop_oldest(out_queue, STREAM_END)  # sentinel must never block either

def collect_stream(out_queue: "queue_mod.Queue") -> np.ndarray:
    chunks: list[np.ndarray] = []

    while True:
        item = out_queue.get()

        if item is STREAM_END:
            break

        chunks.append(item.squeeze(0).cpu().numpy())  # (chunk_frames, 263), batch size 1

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 263), dtype=np.float32)