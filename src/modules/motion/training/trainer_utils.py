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
        # Optimizer state restore: safe; carries running averages.
        if "optimizer_state_dict" in ck:
            try:
                trainer.optimizer.load_state_dict(ck["optimizer_state_dict"])
            except (ValueError, RuntimeError, KeyError):
                log.warning("[BaseTrainer] could not restore optimizer_state_dict")

        # Scheduler state restore: ONLY when the new run targets the same
        # total_steps. OneCycleLR raises if its loaded counter exceeds the
        # newly-constructed total_steps (e.g. resume with --epochs N where
        # N doesn't add new steps beyond what was already taken). In that
        # case keep the freshly-built scheduler -- LR continues from the
        # new schedule's current position, which is the safer default.
        if "scheduler_state_dict" in ck:
            saved_state = ck["scheduler_state_dict"]
            new_total = getattr(trainer.scheduler, "total_steps", None)
            saved_total = saved_state.get("total_steps")

            if (new_total is None or saved_total is None
                    or new_total == saved_total):
                try:
                    trainer.scheduler.load_state_dict(saved_state)
                except (ValueError, RuntimeError, KeyError):
                    log.warning("[BaseTrainer] could not restore scheduler_state_dict")
            else:
                log.warning(
                    "[BaseTrainer] scheduler total_steps changed (%s -> %s); "
                    "keeping freshly-built scheduler",
                    saved_total, new_total,
                )
        trainer.step = ck.get("global_step", 0)
        trainer.start_epoch = ck.get("epoch", 0) + 1
    # best_loss is inherited in both modes so warm-start runs only save a "best"
    # checkpoint when they actually beat the prior baseline.
    trainer.best_loss = ck.get("val_loss", float("inf"))

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


def soft_decode_logits(logits: torch.Tensor, tokenizer) -> torch.Tensor:
    """Differentiable soft codebook lookup + RVQ decode.

    logits:   (B, T', K, V) raw model output over codebook entries
    returns:  (B, T, motion_dim) motion-space prediction whose gradient
              flows back to ``logits`` through the (frozen) decoder.

    Replaces the standard ``tokenizer.decode(argmax(logits))`` path -- which
    is wrapped in ``@torch.no_grad`` and uses a non-differentiable argmax --
    with a softmax-weighted sum over codebook embeddings. The tokenizer's
    decoder weights are frozen but still differentiable forward; gradients
    pass through them to the SSM logits without updating the decoder.
    """
    probs = F.softmax(logits, dim=-1)  # (B, T', K, V)

    soft_z = None
    for k, cb_module in enumerate(tokenizer.rvq.codebooks):
        # cb_module.codebook: (V, latent_dim) -- nn.Parameter, frozen by tokenizer.eval()
        # but still part of the autograd graph for incoming activations.
        embedding = cb_module.codebook  # (V, D)
        # (B, T', V) @ (V, D) -> (B, T', D); sum across K residual layers
        contribution = probs[..., k, :] @ embedding
        soft_z = contribution if soft_z is None else soft_z + contribution

    # decoder expects (B, D, T'); returns (B, motion_dim, T)
    motion_pred = tokenizer.decoder(soft_z.transpose(1, 2)).transpose(1, 2)

    return motion_pred


def geometric_losses(
    logits: torch.Tensor,
    gt_motion: torch.Tensor,
    frame_mask: torch.Tensor,
    tokenizer,
) -> dict[str, torch.Tensor]:
    """MDM-family geometric losses on motion-space (the 'physics-constrained' signal).

    Returns a dict with three terms; the caller multiplies each by its weight.

    * ``recon``       -- L1(soft_motion, gt_motion), masked by frame_mask.
    * ``velocity``    -- L1(diff_t(soft_motion), diff_t(gt_motion)). Penalises
                          jitter; implicit smoothness/physics signal.
    * ``root_height`` -- L1 on channel 5 (transl_z, vertical translation).
                          Cheap surrogate for foot-contact without requiring
                          SMPL-X forward kinematics in the training loop.

    All three are scalar tensors that participate in autograd. Returning
    zeros (not None) when the sequence is too short keeps the graph
    well-formed under masking edge-cases.
    """
    soft_motion = soft_decode_logits(logits, tokenizer)

    # Align T (the decoder may emit a slightly different T than gt_motion):
    T = min(soft_motion.shape[1], gt_motion.shape[1], frame_mask.shape[1])
    soft_motion = soft_motion[:, :T]
    gt_motion = gt_motion[:, :T]
    mask = frame_mask[:, :T].unsqueeze(-1)  # (B, T, 1) broadcast over motion_dim

    # ---- Reconstruction (L1, masked-mean) ----
    abs_err = (soft_motion - gt_motion).abs() * mask
    denom = mask.sum().clamp(min=1) * soft_motion.shape[-1]
    recon = abs_err.sum() / denom

    # ---- Velocity smoothness (L1 on temporal diff) ----
    if T >= 2:
        vel_pred = soft_motion[:, 1:] - soft_motion[:, :-1]
        vel_gt = gt_motion[:, 1:] - gt_motion[:, :-1]
        vel_mask = mask[:, 1:] * mask[:, :-1]  # both endpoints valid
        v_err = (vel_pred - vel_gt).abs() * vel_mask
        v_denom = vel_mask.sum().clamp(min=1) * soft_motion.shape[-1]
        velocity = v_err.sum() / v_denom
    else:
        velocity = soft_motion.new_zeros(())

    # ---- Root vertical drift (channel 5 = transl_z; see src/shared/constants.py) ----
    root_z_pred = soft_motion[..., 5:6]
    root_z_gt = gt_motion[..., 5:6]
    rh_err = (root_z_pred - root_z_gt).abs() * mask[..., :1]
    rh_denom = mask[..., :1].sum().clamp(min=1)
    root_height = rh_err.sum() / rh_denom

    return {"recon": recon, "velocity": velocity, "root_height": root_height}


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
    total_recon = total_velocity = total_root_h = 0.0
    n_steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    down_t = config.rvq_down_t
    amp_active = scaler is not None
    # Geometric-loss weights (default 0.0 for back-compat with old configs)
    w_recon = float(getattr(config, "recon_loss_weight", 0.0))
    w_vel = float(getattr(config, "velocity_loss_weight", 0.0))
    w_rh = float(getattr(config, "root_height_loss_weight", 0.0))
    use_geom = (w_recon + w_vel + w_rh) > 0.0

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

            if use_geom:
                geom = geometric_losses(logits, mgt, mask, tokenizer)
                loss = (
                    loss
                    + w_recon * geom["recon"]
                    + w_vel * geom["velocity"]
                    + w_rh * geom["root_height"]
                )
                total_recon += geom["recon"].item()
                total_velocity += geom["velocity"].item()
                total_root_h += geom["root_height"].item()

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
        postfix = {"loss": f"{loss.item():.4f}", "tok_ce": f"{tok_loss.item():.4f}"}
        if use_geom:
            postfix["recon"] = f"{total_recon / n_steps:.3f}"
            postfix["vel"] = f"{total_velocity / n_steps:.3f}"
        pbar.set_postfix(postfix)

    n = max(len(loader), 1)
    results = (
        total_loss / n,
        total_token_loss / n,
        total_length / n,
        total_recon / n,
        total_velocity / n,
        total_root_h / n,
    )
    return results, n_steps


@torch.no_grad()
def eval_loop(
    model, tokenizer, loader, device, config
) -> tuple[float, float, float, float, float]:
    """Shared eval loop: token CE, top-1/5/10 accuracy, per-codebook top-1.

    Returns:
        (ce, top1_acc, top5_acc, top10_acc, per_cb_top1_acc)
    """
    model.eval()
    total_loss = total_top1 = total_top5 = total_top10 = total_per_cb_acc = 0.0
    n = 0
    down_t = config.rvq_down_t

    for batch in loader:
        inputs = batch_inputs(batch, config, device)
        mgt = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)

        target_tokens = encode_motion_to_tokens(tokenizer, mgt)
        latent_mask = build_latent_mask(mask, down_t)
        ar_targets = (
            target_tokens if getattr(config, "arch", "independent") == "residual_k" else None
        )

        logits, _ = model(inputs, mgt.shape[1], target_tokens=ar_targets)
        t_len = min(logits.shape[1], target_tokens.shape[1])
        logits = logits[:, :t_len]
        target_tokens = target_tokens[:, :t_len]
        latent_mask = latent_mask[:, :t_len]

        total_loss += token_ce_loss(logits, target_tokens, latent_mask).item()

        # Top-k accuracy for k in {1, 5, 10} -- single topk(10) call for all three.
        # logits: (B, T', K, V); target_tokens: (B, T', K)
        k_max = min(10, logits.shape[-1])
        topk_preds = logits.topk(k_max, dim=-1).indices  # (B, T', K, k_max)
        target_exp = target_tokens.unsqueeze(-1)          # (B, T', K, 1)
        valid_sum = latent_mask.sum().clamp(min=1).item()

        for k, attr in zip((1, 5, 10), ("total_top1", "total_top5", "total_top10")):
            if k > k_max:
                break
            hit = (topk_preds[..., :k] == target_exp).any(dim=-1).float()  # (B, T', K)
            acc_k = (hit.mean(dim=-1) * latent_mask).sum().item() / valid_sum
            if attr == "total_top1":
                total_top1 += acc_k
            elif attr == "total_top5":
                total_top5 += acc_k
            else:
                total_top10 += acc_k

        # per-codebook top-1
        correct1 = (topk_preds[..., :1] == target_exp).any(dim=-1).float()  # (B, T', K)
        per_cb = (
            (correct1 * latent_mask.unsqueeze(-1)).sum(dim=(0, 1))
            / latent_mask.sum().clamp(min=1)
        )
        total_per_cb_acc += per_cb.mean().item()
        n += 1

    n = max(n, 1)
    return total_loss / n, total_top1 / n, total_top5 / n, total_top10 / n, total_per_cb_acc / n


@torch.no_grad()
def run_validate(model, tokenizer, loader, device, config) -> tuple:
    """Token cross-entropy + top-1/5/10 accuracy on validation set."""
    ce, top1, top5, top10, _ = eval_loop(model, tokenizer, loader, device, config)
    return ce, top1, top5, top10


@torch.no_grad()
def run_test(model, tokenizer, loader, device, config) -> dict:
    """Final held-out test evaluation — call once after training is complete."""
    ce, top1, top5, top10, cb_acc = eval_loop(model, tokenizer, loader, device, config)
    return {
        "test/ce": ce,
        "test/top1_acc": top1,
        "test/top5_acc": top5,
        "test/top10_acc": top10,
        "test/per_cb_acc": cb_acc,
    }
