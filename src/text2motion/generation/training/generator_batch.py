from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from text2motion.generation.contracts import GeneratorTrainingConfig, MixedPrecisionMode
from text2motion.generation.losses import LossBatch, LossSetup, generator_loss
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.motion.kinematics import Skeleton
from text2motion.motion.representation import KINEMATIC_CHAINS, RAW_OFFSETS

if TYPE_CHECKING:
    from text2motion.generation.training.trainer import GeneratorTrainer


def build_kinematic_context(
    token_generator: MotionTokenGenerator,
    training_config: GeneratorTrainingConfig,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, Skeleton | None]:
    if not training_config.loss_weights.needs_positions:
        return None, None, None
    if mean is None or std is None:
        raise ValueError(
            "FK-consistency and dynamics losses need mean/std (pass them to GeneratorTrainer)"
        )
    device = next(token_generator.parameters()).device
    mean_tensor = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(device)
    std_tensor = torch.from_numpy(np.asarray(std, dtype=np.float32)).to(device)
    skeleton = Skeleton(torch.tensor(RAW_OFFSETS), KINEMATIC_CHAINS, str(device))
    return mean_tensor, std_tensor, skeleton


############################################################
#       GeneratorBatch
############################################################
class GeneratorBatch:
    def __init__(
        self, trainer: GeneratorTrainer, mean: np.ndarray | None, std: np.ndarray | None
    ) -> None:
        self.trainer = trainer
        self._mean, self._std, self._skeleton = build_kinematic_context(
            trainer.token_generator, trainer.training_config, mean, std
        )

    def encode(self, texts: list[str]) -> torch.Tensor:
        if self.trainer.text_encoder is None:
            raise RuntimeError("trainer has no text_encoder; pass a precomputed text_emb instead")
        return self.trainer.text_encoder(texts)

    def drop_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        if self.trainer.training_config.cfg_dropout <= 0:
            return text_emb
        keep_shape = (text_emb.size(0),) + (1,) * (text_emb.dim() - 1)
        keep = (
            torch.rand(keep_shape, device=text_emb.device)
            >= self.trainer.training_config.cfg_dropout
        ).float()
        return text_emb * keep

    def append_end_targets(
        self, target_tokens: torch.Tensor, lengths: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, t_tokens, _ = target_tokens.shape
        device = target_tokens.device
        end_id = self.trainer.token_generator.end_id
        if end_id is None:
            raise ValueError("the generator has no end token; set use_end_token")
        if lengths is None:
            token_lengths = torch.full((batch,), t_tokens, device=device, dtype=torch.long)
        else:
            token_lengths = (lengths // self.trainer.downsample).clamp(min=1, max=t_tokens)
        pad = torch.full_like(target_tokens[:, :1], end_id)
        target_tokens = torch.cat([target_tokens, pad], dim=1)
        target_tokens[torch.arange(batch, device=device), token_lengths] = end_id
        return target_tokens, token_lengths + 1

    def token_targets(
        self,
        gt_motion: torch.Tensor,
        lengths: torch.Tensor | None,
        with_training_noise: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        with torch.no_grad():
            target_tokens = self.trainer.tokenizer_network.encode(gt_motion)

        token_lengths = None
        if self.trainer.token_generator.end_id is not None:
            target_tokens, token_lengths = self.append_end_targets(target_tokens, lengths)

        input_tokens = target_tokens
        if with_training_noise and self.trainer.training_config.pkeep < 1.0:
            input_tokens = self.corrupt_tokens(target_tokens)
        return target_tokens, token_lengths, input_tokens

    def corrupt_tokens(self, target_tokens: torch.Tensor) -> torch.Tensor:
        shape = target_tokens.shape
        if self.trainer.training_config.pkeep_whole_step:
            shape = (*shape[:-1], 1)
        corrupt = (
            torch.rand(shape, device=target_tokens.device) >= self.trainer.training_config.pkeep
        )
        random_tokens = torch.randint_like(
            target_tokens, self.trainer.token_generator.cfg.codebook_size
        )
        return torch.where(corrupt, random_tokens, target_tokens)

    def loss_parts(
        self,
        gt_motion: torch.Tensor,
        text_emb: torch.Tensor,
        lengths: torch.Tensor | None,
        target_tokens: torch.Tensor,
        token_lengths: torch.Tensor | None,
        input_tokens: torch.Tensor,
        with_training_noise: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        amp_on = self.trainer.training_config.amp == MixedPrecisionMode.BF16 and gt_motion.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            unconditional = self.trainer.training_config.cfg_dropout >= 1.0
            noisy = with_training_noise or unconditional
            condition = self.drop_text(text_emb) if noisy else text_emb
            logits = self.trainer.token_generator(input_tokens, condition)
            return generator_loss(
                LossBatch(
                    logits=logits,
                    target_tokens=target_tokens,
                    gt_motion=gt_motion,
                    lengths=lengths,
                    token_lengths=token_lengths,
                ),
                self.loss_setup(),
            )

    def loss_setup(self) -> LossSetup:
        return LossSetup(
            tokenizer=self.trainer.tokenizer_network,
            weights=self.trainer.training_config.loss_weights,
            downsample=self.trainer.downsample,
            has_end=self.trainer.token_generator.end_id is not None,
            mean=self._mean,
            std=self._std,
            skeleton=self._skeleton,
            weighter=self.trainer.weighter,
        )
