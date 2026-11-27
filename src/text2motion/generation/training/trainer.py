from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from text2motion.generation.contracts import GeneratorTrainingConfig
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.text_encoder import CLIPTextEncoder
from text2motion.generation.training.generator_batch import GeneratorBatch
from text2motion.generation.training.optimizer import (
    build_optimizer,
    build_weighter,
    warmup_cosine_factor,
)
from text2motion.tokenization.network.base import TokenizerNetwork
from text2motion.tokenization.parameter_ema import ExponentialMovingAverage

type CheckpointSink = Callable[[int, float], None]


############################################################
#       CleanCrossEntropyLoaders
############################################################
@dataclass(frozen=True)
class CleanCrossEntropyLoaders:
    train: object
    validation: object


############################################################
#       GeneratorTrainer
############################################################
class GeneratorTrainer:
    def __init__(
        self,
        token_generator: MotionTokenGenerator,
        tokenizer_network: TokenizerNetwork,
        training_config: GeneratorTrainingConfig,
        downsample: int,
        text_encoder: CLIPTextEncoder | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.token_generator = token_generator
        self.tokenizer_network = tokenizer_network.eval().requires_grad_(False)
        self.text_encoder = text_encoder
        self.training_config = training_config
        self.downsample = downsample
        self.weighter = build_weighter(token_generator, training_config)
        self.optimizer, self._clip_params = build_optimizer(
            token_generator, text_encoder, training_config, self.weighter
        )
        self.batch = GeneratorBatch(self, mean, std)
        self.parameter_ema = ExponentialMovingAverage(token_generator, training_config.ema_decay)
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.total_steps: int | None = None
        self._accum_count = 0
        self.current_inputs: dict[str, object] = {}
        self.current_step = 0

    def build_scheduler(self, total_steps: int) -> None:
        self.total_steps = total_steps
        warmup = self.training_config.warmup_steps
        floor = self.training_config.lr_min_ratio

        def lr_factor(step: int) -> float:
            return warmup_cosine_factor(step, warmup, total_steps, floor)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)

    def train_step(
        self,
        gt_motion: torch.Tensor,
        text_emb: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        total, parts = self.batch_loss_parts(gt_motion, text_emb, lengths, with_training_noise=True)

        accum = max(1, self.training_config.grad_accum)
        if self._accum_count == 0:
            self.optimizer.zero_grad()
        (total / accum).backward()
        self._accum_count += 1
        if self._accum_count >= accum:
            self.apply_pending_gradients()

        return parts

    @torch.no_grad()
    def teacher_forced_ce(self, loader, device: str, max_batches: int = 0) -> float:
        generator_was_training = self.token_generator.training
        encoder_was_training = self.text_encoder is not None and self.text_encoder.training
        self.token_generator.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        weighted = 0.0
        clips = 0
        try:
            with self.ema_weights():
                for index, batch in enumerate(loader):
                    if max_batches and index >= max_batches:
                        break
                    text_emb = self.batch.encode(list(batch.captions))
                    _, parts = self.batch_loss_parts(
                        batch.features.to(device),
                        text_emb,
                        batch.lengths.to(device),
                        with_training_noise=False,
                    )
                    weighted += float(parts["ce"]) * len(batch.captions)
                    clips += len(batch.captions)
        finally:
            self.token_generator.train(generator_was_training)
            if self.text_encoder is not None:
                self.text_encoder.train(encoder_was_training)
        if clips == 0:
            raise ValueError("the CE loader gave no batch; check the split and max_batches")
        return weighted / clips

    def batch_loss_parts(
        self,
        gt_motion: torch.Tensor,
        text_emb: torch.Tensor,
        lengths: torch.Tensor | None,
        with_training_noise: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        usable = (gt_motion.size(1) // self.downsample) * self.downsample
        gt_motion = gt_motion[:, :usable]
        if lengths is not None:
            lengths = lengths.clamp(max=usable)
        target_tokens, token_lengths, input_tokens = self.batch.token_targets(
            gt_motion, lengths, with_training_noise
        )
        return self.batch.loss_parts(
            gt_motion,
            text_emb,
            lengths,
            target_tokens,
            token_lengths,
            input_tokens,
            with_training_noise,
        )

    def apply_pending_gradients(self) -> None:
        pending = self._accum_count
        if pending == 0:
            return
        self._accum_count = 0
        accum = max(1, self.training_config.grad_accum)
        if pending < accum:
            for parameter in self._clip_params:
                if parameter.grad is not None:
                    parameter.grad.mul_(accum / pending)
        nn.utils.clip_grad_norm_(
            self._clip_params, self.training_config.grad_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.parameter_ema.update(self.token_generator)

    def train_epoch(self, loader, device: str, heartbeat: Path | None = None) -> dict[str, float]:
        self.token_generator.train()
        if self.text_encoder is not None:
            self.text_encoder.train()

        totals: dict[str, torch.Tensor] = {}
        steps = 0
        for batch in loader:
            self.current_inputs = {
                "features": batch.features,
                "lengths": batch.lengths,
                "captions": list(batch.captions),
            }
            self.current_step = steps
            text_emb = self.batch.encode(list(batch.captions))
            parts = self.train_step(batch.features.to(device), text_emb, batch.lengths.to(device))
            for key, value in parts.items():
                running = totals.get(key)
                totals[key] = value.double() if running is None else running + value.double()
            steps += 1
            if heartbeat is not None and steps % 100 == 0:
                heartbeat.write_text(str(steps), encoding="utf-8")
        self.apply_pending_gradients()
        return {key: value.item() / steps for key, value in totals.items()}

    @contextmanager
    def ema_weights(self):
        self.parameter_ema.copy_to(self.token_generator)
        try:
            yield self.token_generator
        finally:
            self.parameter_ema.restore(self.token_generator)
