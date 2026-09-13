from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from text2motion.generation.contracts import SamplingConfig, TextToMotionGenerationRequest
from text2motion.generation.model import MotionTokenGenerator, seeded_sampling_uniforms
from text2motion.generation.text import CLIPTextEncoder
from text2motion.motion.model import (
    GeneratedMotion,
    MotionClip,
    MotionTokens,
)
from text2motion.tokenization.model import MotionTokenizer


class TextToMotionGenerator:
    def __init__(
        self,
        token_generator: MotionTokenGenerator,
        text_encoder: CLIPTextEncoder,
        tokenizer: MotionTokenizer,
    ) -> None:
        self.token_generator = token_generator
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def eval(self) -> TextToMotionGenerator:
        self.token_generator.eval()
        self.text_encoder.eval()
        self.tokenizer.tokenizer_model.eval()
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
    ) -> torch.Tensor | None:
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
        return torch.stack(steps, dim=1)

    def generate_batch(
        self,
        captions: list[str],
        token_steps: int,
        sampling: SamplingConfig,
        seeds: Sequence[int] | None = None,
    ) -> list[GeneratedMotion] | None:
        indices = self.sample_token_sequences(captions, token_steps, sampling, seeds)
        if indices is None:
            return None
        decoded = self.tokenizer.tokenizer_model.decode(indices)
        return [
            GeneratedMotion(
                prompt=caption,
                tokens=MotionTokens(
                    indices=indices[row],
                    token_count=int(indices.shape[1]),
                    frame_count=int(decoded.shape[1]),
                ),
                motion=MotionClip(
                    features=decoded[row],
                    frame_count=int(decoded.shape[1]),
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
