from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from text2motion.motion.dataset import Split
from text2motion.motion.model import MotionBatch
from text2motion.tokenization.model import TokenizerModule, reconstruction_loss

MetricSink = Callable[[dict[str, object]], None]
Evaluator = Callable[[Split], dict[str, float]]


class Ema:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}
        self._tracked = [
            (param, self.shadow[name])
            for name, param in model.named_parameters()
            if name in self.shadow
        ]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        params = [param for param, _ in self._tracked]
        shadows = [shadow for _, shadow in self._tracked]
        torch._foreach_mul_(shadows, self.decay)
        torch._foreach_add_(shadows, params, alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


@dataclass(frozen=True)
class TokenizerTrainingRequest:
    epochs: int = 50
    window: int = 64
    batch_size: int = 128
    ema_decay: float = 0.99
    eval_every: int = 5
    max_eval_clips: int | None = None
    num_workers: int = 0
    checkpoint_name: str | None = None
    resume: bool = False


class TokenizerTrainer:
    def __init__(
        self,
        module: TokenizerModule,
        lr: float,
        weight_decay: float,
        ema_decay: float,
        commit_beta: float = 0.0,
        grad_clip: float = 1.0,
    ) -> None:
        self.module = module
        self.commit_beta = commit_beta
        self.grad_clip = grad_clip
        self.codebook_size = int(module.codebook_size)
        self.opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema = Ema(module, ema_decay)

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
        recon, indices, commit = self.module(motion)

        recon_loss = reconstruction_loss(recon, motion)
        total = recon_loss + self.commit_beta * commit

        self.opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
        self.opt.step()
        self.ema.update(self.module)

        perplexity, usage_frac = self._code_stats(indices)
        return {
            "recon": recon_loss.item(),
            "total": total.item(),
            "perplexity": perplexity,
            "usage_frac": usage_frac,
            "commit": commit.item(),
        }

    def train_epoch(self, loader, device: str) -> dict[str, float]:
        self.module.train()
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
        self.ema.copy_to(self.module)
        try:
            yield self.module
        finally:
            self.ema.restore(self.module)

    def resume_state(self, epoch: int, best_fid: float) -> dict:
        return {
            "tokenizer": self.module.state_dict(),
            "optimizer": self.opt.state_dict(),
            "ema": self.ema.shadow,
            "epoch": epoch,
            "best_fid": best_fid,
        }

    def load_resume_state(self, state: dict, device: str) -> tuple[int, float]:
        self.module.load_state_dict(state["tokenizer"])
        self.opt.load_state_dict(state["optimizer"])
        self.ema.shadow = {key: value.to(device) for key, value in state["ema"].items()}
        return state["epoch"] + 1, state["best_fid"]


def train_tokenizer(
    trainer: TokenizerTrainer,
    loader,
    request: TokenizerTrainingRequest,
    device: str,
    evaluate: Evaluator,
    checkpoint_path: Path,
    resume_path: Path,
    on_metrics: MetricSink = lambda metrics: None,
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
