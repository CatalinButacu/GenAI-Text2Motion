from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from text2motion.generation.model import MotionGeneratorModule, rollout_uniforms
from text2motion.generation.text import CLIPTextEncoder
from text2motion.motion.model import (
    GeneratedMotion,
    MotionClip,
    MotionTokens,
)
from text2motion.tokenization.model import MotionTokenizer


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.9
    cfg_scale: float = 1.0
    stop_at_end: bool = False


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    token_steps: int
    sampling: SamplingConfig = SamplingConfig()
    chunk_tokens: int = 1


class MotionGenerator:
    def __init__(
        self,
        module: MotionGeneratorModule,
        text_encoder: CLIPTextEncoder,
        tokenizer: MotionTokenizer,
    ) -> None:
        self.module = module
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def eval(self) -> MotionGenerator:
        self.module.eval()
        self.text_encoder.eval()
        self.tokenizer.module.eval()
        return self

    @property
    def downsample(self) -> int:
        return self.tokenizer.downsample

    def token_steps_for(self, frames: int) -> int:
        return max(1, frames // self.downsample)

    def stream_tokens(
        self,
        captions: list[str],
        token_steps: int,
        sampling: SamplingConfig,
        seeds: Sequence[int] | None = None,
    ) -> torch.Tensor | None:
        text_emb = self.text_encoder(captions)
        uniforms = None
        if seeds is not None:
            uniforms = rollout_uniforms(
                seeds, token_steps, self.module.cfg.num_codebooks, text_emb.device
            )
        steps = list(
            self.module.stream(
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
        indices = self.stream_tokens(captions, token_steps, sampling, seeds)
        if indices is None:
            return None
        decoded = self.tokenizer.module.decode(indices)
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

    def generate(self, request: GenerationRequest) -> GeneratedMotion | None:
        batch = self.generate_batch([request.prompt], request.token_steps, request.sampling)
        return None if batch is None else batch[0]

    def token_stream(self, request: GenerationRequest) -> Iterator[torch.Tensor]:
        sampling = request.sampling
        text_emb = self.text_encoder([request.prompt])
        return self.module.stream(
            text_emb,
            request.token_steps,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            cfg_scale=sampling.cfg_scale,
            stop_at_end=sampling.stop_at_end,
        )
