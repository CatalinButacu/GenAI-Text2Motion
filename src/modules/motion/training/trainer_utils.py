"""Free-function helpers used by BaseSSMTrainer.

Under the RVQ framing, the SSM trunk predicts codebook indices via cross-entropy.
The RVQ tokenizer is frozen during SSM training -- it encodes ground-truth poses
into target token sequences and is never updated.

Sections:
    - Checkpoint I/O: load_compatible, save/find/resolve/cleanup
    - Seeding: lock_seed
    - Epoch runners: run_train_epoch, run_validate (token cross-entropy)
    - Optimizer/scheduler factory: create_optimizer_and_scheduler
"""

from __future__ import annotations

import glob
import logging
import os
import random
import re
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.shared.seed import seedAll

log = logging.getLogger(__name__)


def loadCompatible(module: torch.nn.Module, stateDict: dict, name: str) -> bool:
    modelSd = module.state_dict()
    compat: dict = {}
    skipped: list = []

    for k, v in stateDict.items():
        if k not in modelSd:
            skipped.append(f"{k} (unexpected)")
            continue

        if v.shape != modelSd[k].shape:
            skipped.append(f"{k}: ckpt {tuple(v.shape)} != model {tuple(modelSd[k].shape)}")
            continue

        compat[k] = v

    if skipped:
        log.warning(
            "[%s] Skipped %d/%d checkpoint keys: %s",
            name,
            len(skipped),
            len(stateDict),
            "; ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""),
        )

    if compat:
        module.load_state_dict(compat, strict=False)
        log.info("[%s] Loaded %d/%d weights from checkpoint", name, len(compat), len(stateDict))

        return True

    log.warning("[%s] No compatible weights in checkpoint - using random init", name)

    return False


def lockSeed(seed: int) -> None:
    """Backward-compat alias. Prefer ``from src.shared.seed import seedAll`` directly."""
    seedAll(seed, deterministic=False)


def saveCkpt(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    log.debug("[BaseTrainer] checkpoint saved -> %s", path)


def findLatestCkpt(ckDir: str) -> str | None:
    files = glob.glob(os.path.join(ckDir, "checkpoint_epoch*.pt"))
    best, best_epoch = None, -1
    rx = re.compile(r"checkpoint_epoch(\d+)\.pt$")

    for f in files:
        if m := rx.search(f):
            e = int(m.group(1))

            if e > best_epoch:
                best_epoch, best = e, f

    return best


def resolveCkptPath(resumeFrom: str, ckDir: str) -> str | None:
    if resumeFrom == "latest":
        p = findLatestCkpt(ckDir)

        if p is None:
            log.warning("[BaseTrainer] no checkpoint found in %s to resume from", ckDir)

        return p

    if os.path.exists(resumeFrom):
        return resumeFrom

    cand = os.path.join(ckDir, resumeFrom)

    if os.path.exists(cand):
        return cand

    log.warning("[BaseTrainer] resume checkpoint %r not found", resumeFrom)

    return None


def cleanupCkpts(ckDir: str, keepLast: int = 5) -> None:
    files = sorted(glob.glob(os.path.join(ckDir, "checkpoint_epoch*.pt")), key=os.path.getmtime)

    for stale in files[:-keepLast]:
        try:
            os.remove(stale)
        except OSError as e:
            log.warning("[BaseTrainer] could not remove %s: %s", stale, e)


def restoreCheckpoint(trainer, path: str, warmStart: bool = False) -> None:
    mode = "warm-start" if warmStart else "full resume"
    log.info("[BaseTrainer] resuming from %s (%s)", path, mode)
    ck = torch.load(path, map_location=trainer.device, weights_only=False)
    loadCompatible(trainer.model, ck["model_state_dict"], "model")

    if not warmStart:
        for key, obj in (
            ("optimizer_state_dict", trainer.optimizer),
            ("scheduler_state_dict", trainer.scheduler),
        ):
            if key in ck:
                try:
                    obj.load_state_dict(ck[key])
                except (ValueError, RuntimeError, KeyError):
                    log.warning("[BaseTrainer] could not restore %s", key)
        trainer.step = ck.get("global_step", 0)
        trainer.startEpoch = ck.get("epoch", 0) + 1
    # bestLoss is inherited in both modes so warm-start runs only save a "best"
    # checkpoint when they actually beat the prior baseline.
    trainer.bestLoss = ck.get("valLoss", ck.get("val_loss", float("inf")))

    if "vocab" in ck and hasattr(trainer, "train_ds"):
        trainer.train_ds.vocab = ck["vocab"]


def createOptimizerAndScheduler(
    params,
    lr: float,
    totalSteps: int,
    warmupSteps: int,
    wd: float = 0.01,
):
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    warmupPct = min(warmupSteps / max(totalSteps, 1), 0.3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=lr,
        total_steps=max(totalSteps, 1),
        pct_start=warmupPct,
    )

    return optim, sched


def batchInputs(batch, config, device):
    return batch["texts"] if config.useSbert else batch["token_ids"].to(device)


@torch.no_grad()
def encodeMotionToTokens(tokenizer, motion: torch.Tensor) -> torch.Tensor:
    """Run the frozen tokenizer to turn GT motion into GT token indices.

    motion: (B, T, motion_dim)  ->  indices: (B, T', K)
    """
    tokenizer.eval()
    return tokenizer.encode(motion)


def tokenCeLoss(
    logits: torch.Tensor, targetTokens: torch.Tensor, latentMask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy over K codebooks.

    logits:        (B, T', K, V)
    targetTokens: (B, T', K)
    latentMask:   (B, T')  -- 1.0 where valid, 0.0 where padded
    """
    B, T, K, V = logits.shape
    flatLogits = logits.reshape(B * T * K, V)
    flatTargets = targetTokens.reshape(B * T * K)
    perTokenLoss = F.cross_entropy(flatLogits, flatTargets, reduction="none")
    perTokenLoss = perTokenLoss.reshape(B, T, K).mean(dim=-1)  # avg across K codebooks

    return (perTokenLoss * latentMask).sum() / latentMask.sum().clamp(min=1)


def buildLatentMask(frameMask: torch.Tensor, downT: int) -> torch.Tensor:
    """Downsample a per-frame mask (B, T) by striding to a per-latent mask (B, T')."""
    return frameMask[:, ::downT]


def applyCfgDropout(inputs, config, device):
    """Classifier-free guidance: randomly replace text with '' during training."""
    p = getattr(config, "cfgDropoutProb", 0.0)

    if p <= 0.0 or not config.useSbert:
        return inputs

    if isinstance(inputs, list):
        return ["" if random.random() < p else t for t in inputs]

    return inputs


def runTrainEpoch(
    model,
    tokenizer,
    loader,
    optimizer,
    scheduler,
    device,
    config,
    epoch: int,
    scaler: Any = None,  # torch.amp.GradScaler | None — left untyped for beartype/forward-ref
) -> tuple:
    """Token cross-entropy training epoch.

    When ``scaler`` is non-None, runs the forward/backward under
    ``torch.amp.autocast('cuda', ...)`` and uses the scaler for loss scaling +
    optimiser step. Caller decides whether AMP is active (CUDA + config.useAmp).
    """
    model.train()
    totalLoss = totalTokenLoss = totalLength = 0.0
    nSteps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    downT = config.rvqDownT
    ampActive = scaler is not None

    for batch in pbar:
        inputs = batchInputs(batch, config, device)
        inputs = applyCfgDropout(inputs, config, device)
        mgt = batch["motion"].to(device)  # (B, T, motion_dim)
        mask = batch["motion_mask"].to(device)  # (B, T)

        # Token encoding stays in fp32 (frozen tokenizer) for stable VQ distances
        targetTokens = encodeMotionToTokens(tokenizer, mgt)  # (B, T', K)
        latentMask = buildLatentMask(mask, downT)

        with torch.amp.autocast(device_type="cuda", enabled=ampActive):  # type: ignore[attr-defined]
            logits, length_pred = model(inputs, mgt.shape[1])  # (B, T', K, V), (B,)
            # Align logits and targets on T' in case of odd trimming
            tLen = min(logits.shape[1], targetTokens.shape[1])
            logits = logits[:, :tLen]
            targetTokensCut = targetTokens[:, :tLen]
            latentMaskCut = latentMask[:, :tLen]

            tokLoss = tokenCeLoss(logits, targetTokensCut, latentMaskCut)
            lenLoss = F.mse_loss(length_pred, batch["length"].float().to(device))
            loss = tokLoss + config.lengthLossWeight * lenLoss

        optimizer.zero_grad()

        if ampActive:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradClip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradClip)
            optimizer.step()
        scheduler.step()

        totalLoss += loss.item()
        totalTokenLoss += tokLoss.item()
        totalLength += lenLoss.item()
        nSteps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "tok_ce": f"{tokLoss.item():.4f}"})

    n = max(len(loader), 1)
    return (totalLoss / n, totalTokenLoss / n, totalLength / n), nSteps


@torch.no_grad()
def evalLoop(model, tokenizer, loader, device, config) -> tuple[float, float, float]:
    """Shared eval loop: token CE, top-1 accuracy, per-codebook top-1."""
    model.eval()
    totalLoss = totalAcc = totalPerCbAcc = 0.0
    n = 0
    downT = config.rvqDownT

    for batch in loader:
        inputs = batchInputs(batch, config, device)
        mgt = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)

        targetTokens = encodeMotionToTokens(tokenizer, mgt)
        latentMask = buildLatentMask(mask, downT)

        logits, _ = model(inputs, mgt.shape[1])
        tLen = min(logits.shape[1], targetTokens.shape[1])
        logits = logits[:, :tLen]
        targetTokens = targetTokens[:, :tLen]
        latentMask = latentMask[:, :tLen]

        totalLoss += tokenCeLoss(logits, targetTokens, latentMask).item()

        pred = logits.argmax(dim=-1)  # (B, T', K)
        correct = (pred == targetTokens).float()
        # top-1 avg across all K codebooks
        totalAcc += (
            (correct.mean(dim=-1) * latentMask).sum().item()
            / latentMask.sum().clamp(min=1).item()
        )
        # per-codebook accuracy (mean across B,T)
        perCb = (correct * latentMask.unsqueeze(-1)).sum(dim=(0, 1)) / latentMask.sum().clamp(min=1)
        totalPerCbAcc += perCb.mean().item()
        n += 1

    n = max(n, 1)
    return totalLoss / n, totalAcc / n, totalPerCbAcc / n


@torch.no_grad()
def runValidate(model, tokenizer, loader, device, config) -> tuple:
    """Token cross-entropy + top-1 accuracy on validation set."""
    ce, acc, _ = evalLoop(model, tokenizer, loader, device, config)
    return ce, acc


@torch.no_grad()
def runTest(model, tokenizer, loader, device, config) -> dict:
    """Final held-out test evaluation — call once after training is complete.

    Returns a dict suitable for JSON export and W&B logging.
    """
    ce, acc, cb_acc = evalLoop(model, tokenizer, loader, device, config)
    return {"test/ce": ce, "test/top1_acc": acc, "test/per_cb_acc": cb_acc}
