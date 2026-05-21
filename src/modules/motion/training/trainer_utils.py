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

from src.shared.seed import seed_all

log = logging.getLogger(__name__)


def load_compatible(module: torch.nn.Module, state_dict: dict, name: str) -> bool:
    model_sd = module.state_dict()
    compat: dict = {}
    skipped: list = []

    for k, v in state_dict.items():
        if k not in model_sd:
            skipped.append(f"{k} (unexpected)")
            continue

        if v.shape != model_sd[k].shape:
            skipped.append(f"{k}: ckpt {tuple(v.shape)} != model {tuple(model_sd[k].shape)}")
            continue

        compat[k] = v

    if skipped:
        log.warning(
            "[%s] Skipped %d/%d checkpoint keys: %s",
            name,
            len(skipped),
            len(state_dict),
            "; ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""),
        )

    if compat:
        module.load_state_dict(compat, strict=False)
        log.info("[%s] Loaded %d/%d weights from checkpoint", name, len(compat), len(state_dict))

        return True

    log.warning("[%s] No compatible weights in checkpoint - using random init", name)

    return False


def lock_seed(seed: int) -> None:
    """Backward-compat alias. Prefer ``from src.shared.seed import seed_all`` directly."""
    seed_all(seed, deterministic=False)


def save_ckpt(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    log.debug("[BaseTrainer] checkpoint saved -> %s", path)


def find_latest_ckpt(ck_dir: str) -> str | None:
    files = glob.glob(os.path.join(ck_dir, "checkpoint_epoch*.pt"))
    best, best_epoch = None, -1
    rx = re.compile(r"checkpoint_epoch(\d+)\.pt$")

    for f in files:
        if m := rx.search(f):
            e = int(m.group(1))

            if e > best_epoch:
                best_epoch, best = e, f

    return best


def resolve_ckpt_path(resume_from: str, ck_dir: str) -> str | None:
    if resume_from == "latest":
        p = find_latest_ckpt(ck_dir)

        if p is None:
            log.warning("[BaseTrainer] no checkpoint found in %s to resume from", ck_dir)

        return p

    if os.path.exists(resume_from):
        return resume_from

    cand = os.path.join(ck_dir, resume_from)

    if os.path.exists(cand):
        return cand

    log.warning("[BaseTrainer] resume checkpoint %r not found", resume_from)

    return None


def cleanup_ckpts(ck_dir: str, keep_last: int = 5) -> None:
    files = sorted(glob.glob(os.path.join(ck_dir, "checkpoint_epoch*.pt")), key=os.path.getmtime)

    for stale in files[:-keep_last]:
        try:
            os.remove(stale)
        except OSError as e:
            log.warning("[BaseTrainer] could not remove %s: %s", stale, e)


def restore_checkpoint(trainer, path: str, warm_start: bool = False) -> None:
    mode = "warm-start" if warm_start else "full resume"
    log.info("[BaseTrainer] resuming from %s (%s)", path, mode)
    ck = torch.load(path, map_location=trainer.device, weights_only=False)
    load_compatible(trainer.model, ck["model_state_dict"], "model")

    if not warm_start:
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
        trainer.start_epoch = ck.get("epoch", 0) + 1
    # best_loss is inherited in both modes so warm-start runs only save a "best"
    # checkpoint when they actually beat the prior baseline.
    trainer.best_loss = ck.get("val_loss", ck.get("val_loss", float("inf")))

    if "vocab" in ck and hasattr(trainer, "train_ds"):
        trainer.train_ds.vocab = ck["vocab"]


def create_optimizer_and_scheduler(
    params,
    lr: float,
    total_steps: int,
    warmup_steps: int,
    wd: float = 0.01,
):
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    warmup_pct = min(warmup_steps / max(total_steps, 1), 0.3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=lr,
        total_steps=max(total_steps, 1),
        pct_start=warmup_pct,
    )

    return optim, sched


def batch_inputs(batch, config, device):
    return batch["texts"] if config.use_sbert else batch["token_ids"].to(device)


@torch.no_grad()
def encode_motion_to_tokens(tokenizer, motion: torch.Tensor) -> torch.Tensor:
    """Run the frozen tokenizer to turn GT motion into GT token indices.

    motion: (B, T, motion_dim)  ->  indices: (B, T', K)
    """
    tokenizer.eval()
    return tokenizer.encode(motion)


def token_ce_loss(
    logits: torch.Tensor, target_tokens: torch.Tensor, latent_mask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy over K codebooks.

    logits:        (B, T', K, V)
    target_tokens: (B, T', K)
    latent_mask:   (B, T')  -- 1.0 where valid, 0.0 where padded
    """
    B, T, K, V = logits.shape
    flat_logits = logits.reshape(B * T * K, V)
    flat_targets = target_tokens.reshape(B * T * K)
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    per_token_loss = per_token_loss.reshape(B, T, K).mean(dim=-1)  # avg across K codebooks

    return (per_token_loss * latent_mask).sum() / latent_mask.sum().clamp(min=1)


def build_latent_mask(frame_mask: torch.Tensor, down_t: int) -> torch.Tensor:
    """Downsample a per-frame mask (B, T) by striding to a per-latent mask (B, T').

    Caller is responsible for ensuring T % down_t == 0 (typically by setting
    max_motion_length as a multiple of down_t). We assert it here so a future
    config change that breaks the invariant fails loud at training time
    instead of producing silently misaligned loss masks.
    """
    T = frame_mask.shape[1]
    assert T % down_t == 0, (
        f"frame mask length {T} not divisible by rvq_down_t={down_t}; "
        f"set max_motion_length to a multiple of {down_t}"
    )
    return frame_mask[:, ::down_t]


def apply_cfg_dropout(inputs, config, device):
    """Classifier-free guidance: randomly replace text with '' during training."""
    p = getattr(config, "cfg_dropout_prob", 0.0)

    if p <= 0.0 or not config.use_sbert:
        return inputs

    if isinstance(inputs, list):
        return ["" if random.random() < p else t for t in inputs]

    return inputs


def run_train_epoch(
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
    optimiser step. Caller decides whether AMP is active (CUDA + config.use_amp).
    """
    model.train()
    total_loss = total_token_loss = total_length = 0.0
    n_steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    down_t = config.rvq_down_t
    amp_active = scaler is not None

    for batch in pbar:
        inputs = batch_inputs(batch, config, device)
        inputs = apply_cfg_dropout(inputs, config, device)
        mgt = batch["motion"].to(device)  # (B, T, motion_dim)
        mask = batch["motion_mask"].to(device)  # (B, T)

        # Token encoding stays in fp32 (frozen tokenizer) for stable VQ distances
        target_tokens = encode_motion_to_tokens(tokenizer, mgt)  # (B, T', K)
        latent_mask = build_latent_mask(mask, down_t)
        # Teacher-force target tokens through the model only when the AR
        # head needs them; the legacy independent head ignores the kwarg.
        ar_targets = (
            target_tokens if getattr(config, "arch", "independent") == "residual_k" else None
        )

        with torch.amp.autocast(device_type="cuda", enabled=amp_active):  # type: ignore[attr-defined]
            logits, length_pred = model(inputs, mgt.shape[1], target_tokens=ar_targets)
            # Align logits and targets on T' in case of odd trimming
            t_len = min(logits.shape[1], target_tokens.shape[1])
            logits = logits[:, :t_len]
            target_tokens_cut = target_tokens[:, :t_len]
            latent_mask_cut = latent_mask[:, :t_len]

            tok_loss = token_ce_loss(logits, target_tokens_cut, latent_mask_cut)
            len_loss = F.mse_loss(length_pred, batch["length"].float().to(device))
            loss = tok_loss + config.length_loss_weight * len_loss

        optimizer.zero_grad()

        if amp_active:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_token_loss += tok_loss.item()
        total_length += len_loss.item()
        n_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "tok_ce": f"{tok_loss.item():.4f}"})

    n = max(len(loader), 1)
    return (total_loss / n, total_token_loss / n, total_length / n), n_steps


@torch.no_grad()
def eval_loop(model, tokenizer, loader, device, config) -> tuple[float, float, float]:
    """Shared eval loop: token CE, top-1 accuracy, per-codebook top-1."""
    model.eval()
    total_loss = total_acc = total_per_cb_acc = 0.0
    n = 0
    down_t = config.rvq_down_t

    for batch in loader:
        inputs = batch_inputs(batch, config, device)
        mgt = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)

        target_tokens = encode_motion_to_tokens(tokenizer, mgt)
        latent_mask = build_latent_mask(mask, down_t)
        # Teacher-force AR head during validation too -- keeps val_ce curves
        # directly comparable to train_ce. Inference uses the AR sampler
        # in ssm_model.py, not this code path.
        ar_targets = (
            target_tokens if getattr(config, "arch", "independent") == "residual_k" else None
        )

        logits, _ = model(inputs, mgt.shape[1], target_tokens=ar_targets)
        t_len = min(logits.shape[1], target_tokens.shape[1])
        logits = logits[:, :t_len]
        target_tokens = target_tokens[:, :t_len]
        latent_mask = latent_mask[:, :t_len]

        total_loss += token_ce_loss(logits, target_tokens, latent_mask).item()

        pred = logits.argmax(dim=-1)  # (B, T', K)
        correct = (pred == target_tokens).float()
        # top-1 avg across all K codebooks
        total_acc += (
            (correct.mean(dim=-1) * latent_mask).sum().item()
            / latent_mask.sum().clamp(min=1).item()
        )
        # per-codebook accuracy (mean across B,T)
        per_cb = (
            (correct * latent_mask.unsqueeze(-1)).sum(dim=(0, 1))
            / latent_mask.sum().clamp(min=1)
        )
        total_per_cb_acc += per_cb.mean().item()
        n += 1

    n = max(n, 1)
    return total_loss / n, total_acc / n, total_per_cb_acc / n


@torch.no_grad()
def run_validate(model, tokenizer, loader, device, config) -> tuple:
    """Token cross-entropy + top-1 accuracy on validation set."""
    ce, acc, _ = eval_loop(model, tokenizer, loader, device, config)
    return ce, acc


@torch.no_grad()
def run_test(model, tokenizer, loader, device, config) -> dict:
    """Final held-out test evaluation — call once after training is complete.

    Returns a dict suitable for JSON export and W&B logging.
    """
    ce, acc, cb_acc = eval_loop(model, tokenizer, loader, device, config)
    return {"test/ce": ce, "test/top1_acc": acc, "test/per_cb_acc": cb_acc}
