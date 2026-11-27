from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from text2motion.motion.contracts import MotionBatch
from text2motion.motion.storage import (
    RESUME_VERSION,
    capture_rng_state,
    restore_rng_state,
)
from text2motion.tokenization.network.base import TokenizerNetwork
from text2motion.tokenization.network.losses import (
    feature_velocity_reconstruction_loss,
    heading_reconstruction_loss,
)
from text2motion.tokenization.parameter_ema import ExponentialMovingAverage

logger = logging.getLogger(__name__)


def pad_to_multiple(motion: torch.Tensor, multiple: int) -> torch.Tensor:
    remainder = motion.shape[1] % multiple
    if remainder == 0:
        return motion
    return torch.nn.functional.pad(motion, (0, 0, 0, multiple - remainder))


TOKENIZER_RESUME_FORMAT = "text2motion.tokenizer_resume"


############################################################
#       TokenizerTrainer
############################################################
class TokenizerTrainer:
    def __init__(
        self,
        tokenizer_network: TokenizerNetwork,
        lr: float,
        weight_decay: float,
        ema_decay: float,
        commitment_beta: float = 0.0,
        grad_clip: float = 1.0,
    ) -> None:
        self.tokenizer_network = tokenizer_network
        self.commitment_beta = commitment_beta
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(
            tokenizer_network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.parameter_ema = ExponentialMovingAverage(
            tokenizer_network, ema_decay, average_buffers=True
        )
        self.current_inputs: dict[str, torch.Tensor] = {}
        self.current_step = 0
        self.steps_taken = 0

    def code_statistics(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codebook_size = self.tokenizer_network.codebook_size
        flat = indices.detach().reshape(-1, indices.shape[-1])
        codebook_count = flat.shape[1]
        dropped_bin = codebook_count * codebook_size
        offsets = torch.arange(codebook_count, device=flat.device) * codebook_size
        bins = torch.where(flat >= 0, flat + offsets, dropped_bin)
        counts = torch.bincount(bins.reshape(-1), minlength=dropped_bin + 1)[:dropped_bin]
        counts = counts.view(codebook_count, codebook_size).float()
        totals = counts.sum(1)
        active = (totals > 0).float()
        probabilities = counts / totals.clamp(min=1).unsqueeze(1)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(1)
        used_fraction = (counts > 0).float().mean(1)
        active_count = active.sum()
        perplexity = (entropy.exp() * active).sum() / active_count
        usage_fraction = (used_fraction * active).sum() / active_count
        return perplexity, usage_fraction

    def objective(
        self,
        reconstruction: torch.Tensor,
        motion: torch.Tensor,
        lengths: torch.Tensor | None,
        commitment_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction_loss = feature_velocity_reconstruction_loss(reconstruction, motion, lengths)
        heading_loss = heading_reconstruction_loss(reconstruction, motion, lengths)
        total = (
            reconstruction_loss
            + self.commitment_beta * commitment_loss
            + self.tokenizer_network.heading_weight * heading_loss
        )
        return reconstruction_loss, total

    def apply_update(self, total: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        total.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.tokenizer_network.parameters(), self.grad_clip
        )
        if not bool(torch.isfinite(gradient_norm) & torch.isfinite(total)):
            raise RuntimeError("non-finite loss or gradient norm; the step is not applied")
        self.optimizer.step()
        self.parameter_ema.update(self.tokenizer_network)

    def train_step(
        self, motion: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        reconstruction, indices, commitment_loss = self.tokenizer_network(motion)
        reconstruction_loss, total = self.objective(
            reconstruction, motion, lengths, commitment_loss
        )
        self.apply_update(total)
        perplexity, usage_fraction = self.code_statistics(indices)

        return {
            "recon": reconstruction_loss.detach(),
            "total": total.detach(),
            "perplexity": perplexity,
            "usage_frac": usage_fraction,
            "commit": commitment_loss.detach(),
        }

    def train_epoch(self, loader, device: str, max_steps: int | None = None) -> dict[str, float]:
        self.tokenizer_network.train()
        totals: dict[str, torch.Tensor] = {}
        steps = 0
        for batch in loader:
            if max_steps is not None and self.steps_taken >= max_steps:
                break
            motion = batch.features if isinstance(batch, MotionBatch) else batch
            motion = pad_to_multiple(motion, self.tokenizer_network.downsample)
            lengths = batch.lengths.to(device) if isinstance(batch, MotionBatch) else None
            self.current_inputs = {"motion": motion}
            self.current_step = steps
            parts = self.train_step(motion.to(device, non_blocking=True), lengths)
            for key, value in parts.items():
                totals[key] = totals[key] + value if key in totals else value
            steps += 1
            self.steps_taken += 1
        if steps == 0:
            raise ValueError("the loader gave no batch; check the split and the window length")
        return {key: value.item() / steps for key, value in totals.items()}

    @contextmanager
    def ema_weights(self):
        self.parameter_ema.copy_to(self.tokenizer_network)
        try:
            yield self.tokenizer_network
        finally:
            self.parameter_ema.restore(self.tokenizer_network)

    def resume_state(self, epoch: int, best_fid: float, run_dir: Path | None) -> dict:
        return {
            "format": TOKENIZER_RESUME_FORMAT,
            "version": RESUME_VERSION,
            "tokenizer": self.tokenizer_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.parameter_ema.shadow,
            "rng": capture_rng_state(),
            "run_dir": None if run_dir is None else str(run_dir),
            "epoch": epoch,
            "best_fid": best_fid,
            "settings": self.resume_settings(),
        }

    def resume_settings(self) -> dict[str, float]:
        return {
            "ema_decay": float(self.parameter_ema.decay),
            "commitment_beta": float(self.commitment_beta),
            "grad_clip": float(self.grad_clip),
        }

    def load_resume_state(self, state: dict, legacy: bool) -> tuple[int, float]:
        stored = state.get("settings")
        if not legacy and stored != self.resume_settings():
            raise ValueError(
                f"the resume file was trained with {stored}, this run uses "
                f"{self.resume_settings()}; resume with the same settings"
            )
        self.tokenizer_network.load_state_dict(state["tokenizer"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.parameter_ema.load_state_dict(state["ema"])

        if legacy:
            logger.warning("old resume file: the RNG state is not in it; it starts fresh")
            return state["epoch"] + 1, state["best_fid"]

        restore_rng_state(state["rng"])
        return state["epoch"] + 1, state["best_fid"]
