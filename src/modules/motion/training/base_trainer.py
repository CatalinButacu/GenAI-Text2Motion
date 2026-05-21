from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.shared.run_ctx import wandb_finish, wandb_log

from ..nn_models import TextToMotionSSM
from ..rvq_tokenizer import MotionRVQTokenizer
from .trainer_utils import (
    cleanup_ckpts,
    create_optimizer_and_scheduler,
    load_compatible,
    lock_seed,
    resolve_ckpt_path,
    restore_checkpoint,
    run_test,
    run_train_epoch,
    run_validate,
    save_ckpt,
)

log = logging.getLogger(__name__)


class BaseSSMTrainer:
    """Abstract trainer: owns state (model/tokenizer/optimiser/loaders) and drives the loop.

    Concrete subclasses implement ``save_checkpoint``. Free-function helpers live
    in :mod:`trainer_utils`.
    """

    config: Any
    device: torch.device
    start_epoch: int
    best_loss: float
    step: int
    model: TextToMotionSSM
    tokenizer: MotionRVQTokenizer
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_ds: Any
    val_ds: Any
    test_ds: Any
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None

    lock_seed = staticmethod(lock_seed)

    save_checkpoint_at = staticmethod(save_ckpt)

    # --- init ---------------------------------------------------------------
    def load_frozen_tokenizer(self, config) -> MotionRVQTokenizer:
        path = config.rvq_checkpoint_path

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RVQ tokenizer checkpoint not found at {path!r}. "
                "Train it first with scripts/training/train_rvq_tokenizer.py."
            )
        tok = MotionRVQTokenizer(
            motion_dim=config.motion_dim,
            latent_dim=config.rvq_latent_dim,
            n_codebooks=config.rvq_n_codebooks,
            codebook_size=config.rvq_codebook_size,
            down_t=config.rvq_down_t,
        ).to(self.device)
        ck = torch.load(path, map_location=self.device, weights_only=False)
        load_compatible(tok, ck["model_state_dict"], "rvq_tokenizer")
        tok.eval()

        for p in tok.parameters():
            p.requires_grad = False
        log.info("[BaseTrainer] frozen RVQ tokenizer loaded from %s", path)

        return tok

    def finalize_init(self, config, train_sampler=None) -> None:
        """Shared last step of __init__: build loaders, tokenizer, optimizer, restore ckpt."""
        self.model = TextToMotionSSM(config).to(self.device)
        self.tokenizer = self.load_frozen_tokenizer(config)

        pin = config.num_workers > 0 and self.device.type == "cuda"
        loader_kwargs = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": pin,
        }

        if train_sampler is not None:
            self.train_loader = DataLoader(self.train_ds, sampler=train_sampler, **loader_kwargs)
        else:
            self.train_loader = DataLoader(self.train_ds, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(
            self.test_ds, shuffle=False, **loader_kwargs
        ) if getattr(self, "test_ds", None) is not None else None

        total_steps = max(len(self.train_loader) * config.num_epochs, 1)
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model.parameters(),
            config.learning_rate,
            total_steps,
            config.warmup_steps,
            config.weight_decay,
        )
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.step = 0
        self.best_loss = float("inf")
        self.start_epoch = 1

        amp_on = bool(getattr(config, "use_amp", False)) and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler(device="cuda", enabled=amp_on)  # type: ignore[attr-defined]
            if amp_on else None
        )
        if amp_on:
            log.info("[BaseTrainer] mixed precision (CUDA AMP) enabled")

        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    # --- epoch loop ---------------------------------------------------------
    def should_stop_early(self, no_improve: int, patience: int) -> bool:
        if patience > 0 and no_improve >= patience:
            log.info("[BaseTrainer] early stopping: no improvement for %d epochs", patience)

            return True

        return False

    def train_epoch(self, epoch: int) -> tuple:
        results, n_steps = run_train_epoch(
            self.model,
            self.tokenizer,
            self.train_loader,
            self.optimizer,
            self.scheduler,
            self.device,
            self.config,
            epoch,
            scaler=self.scaler,
        )
        self.step += n_steps

        return results

    def validate(self) -> tuple:
        return run_validate(self.model, self.tokenizer, self.val_loader, self.device, self.config)

    def run_final_test(self) -> dict:
        if self.test_loader is None:
            log.warning("[BaseTrainer] no test loader -- skipping final test evaluation")
            return {}
        metrics = run_test(self.model, self.tokenizer, self.test_loader, self.device, self.config)
        log.info(
            "[BaseTrainer] TEST  ce=%.4f  top1=%.3f  per_cb_acc=%.3f",
            metrics["test/ce"], metrics["test/top1_acc"], metrics["test/per_cb_acc"],
        )
        wandb_log(metrics)
        return metrics

    def log_epoch(self, epoch: int, tr: tuple, vr: tuple) -> None:
        tl, tok_ce, len_l = tr
        val_ce, val_acc = vr
        lr = self.optimizer.param_groups[0]["lr"]
        log.info(
            "epoch=%d/%d train=%.4f(ce=%.4f len=%.3f) val_ce=%.4f top1=%.3f lr=%.2e",
            epoch, self.config.num_epochs, tl, tok_ce, len_l, val_ce, val_acc, lr,
        )
        wandb_log({
            "epoch": epoch, "train/loss": tl, "train/tok_ce": tok_ce, "train/len_loss": len_l,
            "val/ce": val_ce, "val/top1": val_acc, "lr": lr, "best_val_loss": self.best_loss,
        }, step=epoch)

    def load_checkpoint(self, resume_from: str) -> None:
        path = resolve_ckpt_path(resume_from, self.config.checkpoint_dir)

        if path:
            warm_start = getattr(self.config, "warm_start", False)
            restore_checkpoint(self, path, warm_start=warm_start)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        raise NotImplementedError

    def train(self) -> float:
        no_improve = 0

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            tr = self.train_epoch(epoch)
            vr = self.validate()
            val_loss = vr[0] if isinstance(vr, tuple) else float(vr)
            self.log_epoch(epoch, tr, vr)
            is_best = val_loss < self.best_loss

            if is_best:
                self.best_loss, no_improve = val_loss, 0
            else:
                no_improve += 1
            save_every = getattr(self.config, "save_every", 10)

            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            keep_last = getattr(self.config, "keep_last_checkpoints", 0)

            if keep_last > 0:
                cleanup_ckpts(self.config.checkpoint_dir, keep_last)

            patience = getattr(self.config, "early_stop_patience", 0)

            if self.should_stop_early(no_improve, patience):
                break

        self.run_final_test()
        wandb_finish()

        return self.best_loss
