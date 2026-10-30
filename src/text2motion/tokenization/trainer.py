from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from text2motion.motion.contracts import Evaluator, MetricSink, Split
from text2motion.motion.model import MotionBatch
from text2motion.motion.storage import (
    RESUME_VERSION,
    atomic_torch_save,
    capture_rng_state,
    dump_on_failure,
    is_legacy_resume,
    restore_rng_state,
)
from text2motion.tokenization.contracts import TrainTokenizerRequest
from text2motion.tokenization.ema import ExponentialMovingAverage
from text2motion.tokenization.nn_model import (
    TokenizerNetwork,
    feature_velocity_reconstruction_loss,
    reject_non_finite_motion,
)

logger = logging.getLogger(__name__)


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

    def train_step(self, motion: torch.Tensor) -> dict[str, torch.Tensor]:
        checked = reject_non_finite_motion(motion)
        reconstruction, indices, commitment_loss = self.tokenizer_network(checked)

        reconstruction_loss = feature_velocity_reconstruction_loss(reconstruction, motion)
        total = reconstruction_loss + self.commitment_beta * commitment_loss

        self.optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(
            self.tokenizer_network.parameters(), self.grad_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.parameter_ema.update(self.tokenizer_network)

        perplexity, usage_fraction = self.code_statistics(indices)

        return {
            "recon": reconstruction_loss.detach(),
            "total": total.detach(),
            "perplexity": perplexity,
            "usage_frac": usage_fraction,
            "commit": commitment_loss.detach(),
        }

    def train_epoch(self, loader, device: str) -> dict[str, float]:
        self.tokenizer_network.train()
        totals: dict[str, torch.Tensor] = {}
        steps = 0
        for batch in loader:
            motion = batch.features if isinstance(batch, MotionBatch) else batch
            self.current_inputs = {"motion": motion}
            self.current_step = steps
            parts = self.train_step(motion.to(device, non_blocking=True))
            for key, value in parts.items():
                totals[key] = totals[key] + value if key in totals else value
            steps += 1
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
        }

    def load_resume_state(self, state: dict, legacy: bool) -> tuple[int, float]:
        self.tokenizer_network.load_state_dict(state["tokenizer"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.parameter_ema.load_state_dict(state["ema"])

        if legacy:
            logger.warning("old resume file: the RNG state is not in it; it starts fresh")
            return state["epoch"] + 1, state["best_fid"]

        restore_rng_state(state["rng"])
        return state["epoch"] + 1, state["best_fid"]


def train_tokenizer(
    trainer: TokenizerTrainer,
    loader,
    request: TrainTokenizerRequest,
    device: str,
    evaluate: Evaluator,
    checkpoint_path: Path,
    resume_path: Path,
    on_metrics: MetricSink = lambda metrics: None,
    run_dir: Path | None = None,
    failure_dump_path: Path | None = None,
) -> float:
    best_fid = float("inf")
    start_epoch = 0

    if request.resume and not resume_path.is_file():
        raise FileNotFoundError(
            f"--resume continues from {resume_path}, which does not exist; start without --resume"
        )
    if request.resume:
        state = torch.load(resume_path, map_location=device, weights_only=True)
        legacy = is_legacy_resume(
            state, resume_path, TOKENIZER_RESUME_FORMAT, request.accept_legacy_resume
        )
        start_epoch, best_fid = trainer.load_resume_state(state, legacy)
        logger.info(
            f"resumed from {resume_path} at epoch {start_epoch} (best recon-FID {best_fid:.4f})"
        )

    for epoch in range(start_epoch, request.epochs):
        loader.dataset.set_epoch(epoch)
        with dump_on_failure(
            failure_dump_path,
            lambda epoch=epoch, best_fid=best_fid: (
                trainer.resume_state(epoch, best_fid, run_dir),
                trainer.current_inputs,
                trainer.current_step,
                trainer.tokenizer_network,
            ),
        ):
            means = trainer.train_epoch(loader, device)
        on_metrics({"epoch": epoch + 1, **means})
        logger.info(
            f"epoch {epoch + 1:3d}  [common] recon {means['recon']:.4f} total {means['total']:.4f}  "
            f"[health] perplexity {means['perplexity']:.1f} usage {means['usage_frac']:.1%} "
            f"commit {means['commit']:.4f}"
        )

        if (epoch + 1) % request.eval_every and epoch + 1 != request.epochs:
            continue

        with trainer.ema_weights():
            metrics = evaluate(Split.VALIDATION)
        on_metrics({"epoch": epoch + 1, "split": Split.VALIDATION.value, **metrics})
        logger.info(
            f"  [val] clips {metrics['clips']}  MPJPE {metrics['mpjpe_mm']:.1f}mm  "
            f"feat-L2 {metrics['feature_l2']:.4f}  recon-FID {metrics['recon_fid']:.4f}"
        )

        not_finite = sorted(name for name, value in metrics.items() if not math.isfinite(value))
        if not_finite:
            raise RuntimeError(f"epoch {epoch + 1}: evaluation metrics not finite: {not_finite}")
        if metrics["recon_fid"] < best_fid:
            best_fid = metrics["recon_fid"]
            with trainer.ema_weights() as module:
                atomic_torch_save(module.state_dict(), checkpoint_path)
            logger.info(f"  saved best (val) -> {checkpoint_path} (val recon-FID {best_fid:.4f})")

        atomic_torch_save(trainer.resume_state(epoch, best_fid, run_dir), resume_path)

    return best_fid
