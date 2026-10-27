from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from text2motion.generation.contracts import SamplingConfig, TextToMotionGenerationRequest
from text2motion.generation.nn_model import MotionTokenGenerator, seeded_sampling_uniforms
from text2motion.generation.nn_text import CLIPTextEncoder
from text2motion.motion.model import (
    GeneratedMotion,
    MotionClip,
    MotionTokens,
)
from text2motion.tokenization.nn_model import FrozenMotionTokenizer


class TextToMotionGenerator:
    def __init__(
        self,
        token_generator: MotionTokenGenerator,
        text_encoder: CLIPTextEncoder,
        tokenizer: FrozenMotionTokenizer,
    ) -> None:
        self.token_generator = token_generator
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def eval(self) -> TextToMotionGenerator:
        self.token_generator.eval()
        self.text_encoder.eval()
        self.tokenizer.network.eval()
        return self

    @property
    def downsample(self) -> int:
        return self.tokenizer.downsample

    def token_steps_for(self, frames: int) -> int:
        return max(1, frames // self.downsample)

    def sample_token_sequences(
        self,
        captions: list[str],
        token_steps: int,
        sampling: SamplingConfig,
        seeds: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]] | None:
        text_emb = self.text_encoder(captions)
        uniforms = None
        if seeds is not None:
            uniforms = seeded_sampling_uniforms(
                seeds, token_steps, self.token_generator.cfg.num_codebooks, text_emb.device
            )
        steps = list(
            self.token_generator.stream_token_indices(
                text_emb,
                token_steps,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                cfg_scale=sampling.cfg_scale,
                stop_at_end=sampling.stop_at_end,
                uniforms=uniforms,
            )
        )
        if not steps:
            return None
        indices = torch.stack(steps, dim=1)
        return indices, self.__row_lengths(indices)

    def __row_lengths(self, indices: torch.Tensor) -> list[int]:
        end_id = self.token_generator.end_id
        if end_id is None:
            return [indices.shape[1]] * indices.shape[0]
        ended = (indices == end_id).any(dim=-1)
        first_end = ended.int().argmax(dim=1)
        return torch.where(ended.any(dim=1), first_end, indices.shape[1]).tolist()

    def __decode_rows(self, indices: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
        decoder = self.tokenizer.network
        if len(set(lengths)) == 1:
            return list(decoder.decode(indices[:, : lengths[0]]))
        return [
            decoder.decode(indices[row : row + 1, :length])[0] for row, length in enumerate(lengths)
        ]

    def generate_batch(
        self,
        captions: list[str],
        token_steps: int,
        sampling: SamplingConfig,
        seeds: Sequence[int] | None = None,
    ) -> list[GeneratedMotion] | None:
        sampled = self.sample_token_sequences(captions, token_steps, sampling, seeds)
        if sampled is None:
            return None
        indices, lengths = sampled
        decoded = self.__decode_rows(indices, lengths)
        return [
            GeneratedMotion(
                prompt=caption,
                tokens=MotionTokens(
                    indices=indices[row, : lengths[row]],
                    token_count=lengths[row],
                    frame_count=int(decoded[row].shape[0]),
                ),
                motion=MotionClip(
                    features=decoded[row],
                    frame_count=int(decoded[row].shape[0]),
                    caption=caption,
                ),
            )
            for row, caption in enumerate(captions)
        ]

    def generate(self, request: TextToMotionGenerationRequest) -> GeneratedMotion | None:
        batch = self.generate_batch([request.prompt], request.token_steps, request.sampling)
        return None if batch is None else batch[0]

    def stream_token_indices(
        self, request: TextToMotionGenerationRequest
    ) -> Iterator[torch.Tensor]:
        sampling = request.sampling
        text_emb = self.text_encoder([request.prompt])
        return self.token_generator.stream_token_indices(
            text_emb,
            request.token_steps,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            cfg_scale=sampling.cfg_scale,
            stop_at_end=sampling.stop_at_end,
        )
