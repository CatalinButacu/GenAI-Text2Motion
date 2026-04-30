from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.shared.run_ctx import wandbFinish, wandbLog

from ..nn_models import TextToMotionSSM
from ..rvq_tokenizer import MotionRVQTokenizer
from .trainer_utils import (
    cleanupCkpts,
    createOptimizerAndScheduler,
    loadCompatible,
    lockSeed,
    resolveCkptPath,
    restoreCheckpoint,
    runTest,
    runTrainEpoch,
    runValidate,
    saveCkpt,
)

log = logging.getLogger(__name__)


class BaseSSMTrainer:
    """Abstract trainer: owns state (model/tokenizer/optimiser/loaders) and drives the loop.

    Concrete subclasses implement ``save_checkpoint``. Free-function helpers live
    in :mod:`trainer_utils`.
    """

    config: Any
    device: torch.device
    startEpoch: int
    bestLoss: float
    step: int
    model: TextToMotionSSM
    tokenizer: MotionRVQTokenizer
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    trainDs: Any
    valDs: Any
    testDs: Any
    trainLoader: DataLoader
    valLoader: DataLoader
    testLoader: DataLoader | None

    lockSeed = staticmethod(lockSeed)

    saveCheckpointAt = staticmethod(saveCkpt)

    # --- init ---------------------------------------------------------------
    def loadFrozenTokenizer(self, config) -> MotionRVQTokenizer:
        path = config.rvqCheckpointPath

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RVQ tokenizer checkpoint not found at {path!r}. "
                "Train it first with scripts/training/train_rvq_tokenizer.py."
            )
        tok = MotionRVQTokenizer(
            motionDim=config.motionDim,
            latentDim=config.rvqLatentDim,
            nCodebooks=config.rvqNCodebooks,
            codebookSize=config.rvqCodebookSize,
            downT=config.rvqDownT,
        ).to(self.device)
        ck = torch.load(path, map_location=self.device, weights_only=False)
        loadCompatible(tok, ck["model_state_dict"], "rvq_tokenizer")
        tok.eval()

        for p in tok.parameters():
            p.requires_grad = False
        log.info("[BaseTrainer] frozen RVQ tokenizer loaded from %s", path)

        return tok

    def finalizeInit(self, config, trainSampler=None) -> None:
        """Shared last step of __init__: build loaders, tokenizer, optimizer, restore ckpt."""
        self.model = TextToMotionSSM(config).to(self.device)
        self.tokenizer = self.loadFrozenTokenizer(config)

        pin = config.numWorkers > 0 and self.device.type == "cuda"
        loaderKwargs = {
            "batch_size": config.batchSize,
            "num_workers": config.numWorkers,
            "pin_memory": pin,
        }

        if trainSampler is not None:
            self.trainLoader = DataLoader(self.trainDs, sampler=trainSampler, **loaderKwargs)
        else:
            self.trainLoader = DataLoader(self.trainDs, shuffle=True, **loaderKwargs)
        self.valLoader = DataLoader(self.valDs, shuffle=False, **loaderKwargs)
        self.testLoader = DataLoader(
            self.testDs, shuffle=False, **loaderKwargs
        ) if getattr(self, "testDs", None) is not None else None

        totalSteps = max(len(self.trainLoader) * config.numEpochs, 1)
        self.optimizer, self.scheduler = createOptimizerAndScheduler(
            self.model.parameters(),
            config.learningRate,
            totalSteps,
            config.warmupSteps,
            config.weightDecay,
        )
        os.makedirs(config.checkpointDir, exist_ok=True)
        self.step = 0
        self.bestLoss = float("inf")
        self.startEpoch = 1

        ampOn = bool(getattr(config, "useAmp", False)) and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler(device="cuda", enabled=ampOn)  # type: ignore[attr-defined]
            if ampOn else None
        )
        if ampOn:
            log.info("[BaseTrainer] mixed precision (CUDA AMP) enabled")

        if config.resumeFrom:
            self.loadCheckpoint(config.resumeFrom)

    # --- epoch loop ---------------------------------------------------------
    def shouldStopEarly(self, noImprove: int, patience: int) -> bool:
        if patience > 0 and noImprove >= patience:
            log.info("[BaseTrainer] early stopping: no improvement for %d epochs", patience)

            return True

        return False

    def trainEpoch(self, epoch: int) -> tuple:
        results, n_steps = runTrainEpoch(
            self.model,
            self.tokenizer,
            self.trainLoader,
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
        return runValidate(self.model, self.tokenizer, self.valLoader, self.device, self.config)

    def runFinalTest(self) -> dict:
        if self.testLoader is None:
            log.warning("[BaseTrainer] no test loader -- skipping final test evaluation")
            return {}
        metrics = runTest(self.model, self.tokenizer, self.testLoader, self.device, self.config)
        log.info(
            "[BaseTrainer] TEST  ce=%.4f  top1=%.3f  per_cb_acc=%.3f",
            metrics["test/ce"], metrics["test/top1_acc"], metrics["test/per_cb_acc"],
        )
        wandbLog(metrics)
        return metrics

    def logEpoch(self, epoch: int, tr: tuple, vr: tuple) -> None:
        tl, tok_ce, len_l = tr
        val_ce, val_acc = vr
        lr = self.optimizer.param_groups[0]["lr"]
        log.info(
            "epoch=%d/%d train=%.4f(ce=%.4f len=%.3f) val_ce=%.4f top1=%.3f lr=%.2e",
            epoch, self.config.numEpochs, tl, tok_ce, len_l, val_ce, val_acc, lr,
        )
        wandbLog({
            "epoch": epoch, "train/loss": tl, "train/tok_ce": tok_ce, "train/len_loss": len_l,
            "val/ce": val_ce, "val/top1": val_acc, "lr": lr, "best_val_loss": self.bestLoss,
        }, step=epoch)

    def loadCheckpoint(self, resumeFrom: str) -> None:
        path = resolveCkptPath(resumeFrom, self.config.checkpointDir)

        if path:
            restoreCheckpoint(self, path)

    def saveCheckpoint(self, epoch: int, valLoss: float, isBest: bool) -> None:
        raise NotImplementedError

    def train(self) -> float:
        noImprove = 0

        for epoch in range(self.startEpoch, self.config.numEpochs + 1):
            tr = self.trainEpoch(epoch)
            vr = self.validate()
            valLoss = vr[0] if isinstance(vr, tuple) else float(vr)
            self.logEpoch(epoch, tr, vr)
            isBest = valLoss < self.bestLoss

            if isBest:
                self.bestLoss, noImprove = valLoss, 0
            else:
                noImprove += 1
            saveEvery = getattr(self.config, "saveEvery", 10)

            if epoch % saveEvery == 0 or isBest:
                self.saveCheckpoint(epoch, valLoss, isBest)
            keepLast = getattr(self.config, "keepLastCheckpoints", 0)

            if keepLast > 0:
                cleanupCkpts(self.config.checkpointDir, keepLast)

            patience = getattr(self.config, "earlyStopPatience", 0)

            if self.shouldStopEarly(noImprove, patience):
                break

        self.runFinalTest()
        wandbFinish()

        return self.bestLoss
