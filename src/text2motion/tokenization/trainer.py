from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from text2motion.motion.contracts import Split
from text2motion.motion.model import MotionBatch
from text2motion.tokenization.contracts import TokenizerTrainingRequest
from text2motion.tokenization.ema import ExponentialMovingAverage
from text2motion.tokenization.model import (
    MotionTokenizerNetwork,
    feature_velocity_reconstruction_loss,
)

_MetricSink = Callable[[dict[str, object]], None]
_Evaluator = Callable[[Split], dict[str, float]]


class TokenizerTrainer:
    def __init__(
        self,
        tokenizer_network: MotionTokenizerNetwork,
        lr: float,
        weight_decay: float,
        ema_decay: float,
        commit_beta: float = 0.0,
        grad_clip: float = 1.0,
    ) -> None:
        self.tokenizer_network = tokenizer_network
        self.commit_beta = commit_beta
        self.grad_clip = grad_clip
        self.codebook_size = int(tokenizer_network.codebook_size)
        self.optimizer = torch.optim.AdamW(
            tokenizer_network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.parameter_ema = ExponentialMovingAverage(tokenizer_network, ema_decay)

    def _code_stats(self, indices: torch.Tensor) -> tuple[float, float]:
        indices = indices.detach().cpu()
        perplexities, usages = [], []
        for codebook in range(indices.shape[-1]):
            counts = torch.bincount(
                indices[..., codebook].reshape(-1), minlength=self.codebook_size
            ).float()
            probs = counts[counts > 0] / counts.sum()
            perplexities.append(float(torch.exp(-(probs * probs.log()).sum())))
            usages.append(float((counts > 0).float().mean()))
        return sum(perplexities) / len(perplexities), sum(usages) / len(usages)

    def train_step(self, motion: torch.Tensor) -> dict[str, float]:
        recon, indices, commit = self.tokenizer_network(motion)

        recon_loss = feature_velocity_reconstruction_loss(recon, motion)
        total = recon_loss + self.commit_beta * commit

        self.optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.tokenizer_network.parameters(), self.grad_clip)
        self.optimizer.step()
        self.parameter_ema.update(self.tokenizer_network)

        perplexity, usage_frac = self._code_stats(indices)
        return {
            "recon": recon_loss.item(),
            "total": total.item(),
            "perplexity": perplexity,
            "usage_frac": usage_frac,
            "commit": commit.item(),
        }

    def train_epoch(self, loader, device: str) -> dict[str, float]:
        self.tokenizer_network.train()
        totals: dict[str, float] = {}
        steps = 0
        for batch in loader:
            motion = batch.features if isinstance(batch, MotionBatch) else batch
            parts = self.train_step(motion.to(device))
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        return {key: value / steps for key, value in totals.items()}

    @contextmanager
    def ema_weights(self):
        self.parameter_ema.copy_to(self.tokenizer_network)
        try:
            yield self.tokenizer_network
        finally:
            self.parameter_ema.restore(self.tokenizer_network)

    def resume_state(self, epoch: int, best_fid: float) -> dict:
        return {
            "tokenizer": self.tokenizer_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.parameter_ema.shadow,
            "epoch": epoch,
            "best_fid": best_fid,
        }

    def load_resume_state(self, state: dict, device: str) -> tuple[int, float]:
        self.tokenizer_network.load_state_dict(state["tokenizer"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.parameter_ema.shadow = {key: value.to(device) for key, value in state["ema"].items()}
        return state["epoch"] + 1, state["best_fid"]


def train_tokenizer(
    trainer: TokenizerTrainer,
    loader,
    request: TokenizerTrainingRequest,
    device: str,
    evaluate: _Evaluator,
    checkpoint_path: Path,
    resume_path: Path,
    on_metrics: _MetricSink = lambda metrics: None,
) -> float:
    best_fid = float("inf")
    start_epoch = 0

    if request.resume and resume_path.is_file():
        start_epoch, best_fid = trainer.load_resume_state(
            torch.load(resume_path, map_location=device), device
        )
        print(f"resumed from {resume_path} at epoch {start_epoch} (best recon-FID {best_fid:.4f})")

    for epoch in range(start_epoch, request.epochs):
        loader.dataset.set_epoch(epoch)
        means = trainer.train_epoch(loader, device)
        on_metrics({"epoch": epoch + 1, **means})
        print(
            f"epoch {epoch + 1:3d}  [common] recon {means['recon']:.4f} total {means['total']:.4f}  "
            f"[health] perplexity {means['perplexity']:.1f} usage {means['usage_frac']:.1%} "
            f"commit {means['commit']:.4f}"
        )

        if (epoch + 1) % request.eval_every and epoch + 1 != request.epochs:
            continue

        with trainer.ema_weights():
            metrics = evaluate(Split.VALIDATION)
        on_metrics({"epoch": epoch + 1, "split": Split.VALIDATION.value, **metrics})
        print(
            f"  [val] clips {metrics['clips']}  MPJPE {metrics['mpjpe_mm']:.1f}mm  "
            f"feat-L2 {metrics['feature_l2']:.4f}  recon-FID {metrics['recon_fid']:.4f}"
        )

        if metrics["recon_fid"] < best_fid:
            best_fid = metrics["recon_fid"]
            with trainer.ema_weights() as module:
                torch.save(module.state_dict(), checkpoint_path)
            print(f"  saved best (val) -> {checkpoint_path} (val recon-FID {best_fid:.4f})")

        torch.save(trainer.resume_state(epoch, best_fid), resume_path)

    return best_fid
