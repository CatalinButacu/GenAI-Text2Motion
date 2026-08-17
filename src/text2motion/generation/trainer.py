from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from torch import nn

from text2motion.generation.losses import (
    LossWeighting,
    LossWeights,
    UncertaintyWeighter,
    generator_loss,
)
from text2motion.generation.model import MotionGeneratorModule
from text2motion.generation.text import CLIPTextEncoder
from text2motion.motion.dataset import Split
from text2motion.motion.kinematics import Skeleton
from text2motion.motion.representation import t2m_kinematic_chain, t2m_raw_offsets
from text2motion.tokenization.model import TokenizerModule
from text2motion.tokenization.trainer import Ema

NO_DECAY_SUFFIXES = ("a_log", "d_skip")
NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)

MetricSink = Callable[[dict[str, object]], None]
Evaluator = Callable[[Split], dict[str, float]]
CheckpointSink = Callable[[int, float], None]


class Amp(StrEnum):
    OFF = "off"
    BF16 = "bf16"


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 2e-4
    text_encoder_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_min_ratio: float = 0.01
    ema_decay: float = 0.999
    cfg_dropout: float = 0.1
    pkeep: float = 0.8
    amp: Amp = Amp.OFF
    grad_accum: int = 1
    grad_clip: float = 1.0
    decay_groups: bool = True
    loss_weighting: LossWeighting = LossWeighting.FIXED
    loss_weights: LossWeights = field(default_factory=LossWeights)


@dataclass(frozen=True)
class GeneratorTrainingRequest:
    epochs: int = 60
    batch_size: int = 64
    grad_accum: int = 1
    eval_every: int = 5
    eval_split: Split = Split.VALIDATION
    max_eval_clips: int = 600
    temperature: float = 1.0
    cfg_scale: float = 1.0
    checkpoint_name: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class ParameterPartition:
    decay: tuple[nn.Parameter, ...]
    no_decay: tuple[nn.Parameter, ...]

    def optimizer_groups(self, lr: float) -> list[dict]:
        groups: list[dict] = []
        if self.decay:
            groups.append({"params": list(self.decay), "lr": lr})
        if self.no_decay:
            groups.append({"params": list(self.no_decay), "lr": lr, "weight_decay": 0.0})
        return groups

    def assert_complete(self, expected: list[nn.Parameter]) -> None:
        expected_ids = {id(parameter) for parameter in expected if parameter.requires_grad}
        decay_ids = {id(parameter) for parameter in self.decay}
        no_decay_ids = {id(parameter) for parameter in self.no_decay}
        if decay_ids & no_decay_ids:
            raise RuntimeError("optimizer parameter partitions overlap")
        if decay_ids | no_decay_ids != expected_ids:
            raise RuntimeError("optimizer parameter partitions do not cover trainable parameters")


def _undecayed_parameter_ids(module: nn.Module) -> set[int]:
    ids: set[int] = set()
    for submodule in module.modules():
        if isinstance(submodule, (nn.Embedding, *NORM_TYPES)):
            ids.update(id(parameter) for parameter in submodule.parameters(recurse=False))
        elif type(submodule).__name__ == "RMSNorm":
            ids.update(id(parameter) for parameter in submodule.parameters(recurse=False))
    return ids


def partition_parameters(
    module: nn.Module, only: list[nn.Parameter] | None = None
) -> ParameterPartition:
    allowed = None if only is None else {id(parameter) for parameter in only}
    exempt = _undecayed_parameter_ids(module)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or (allowed is not None and id(parameter) not in allowed):
            continue
        if parameter.ndim <= 1 or id(parameter) in exempt or name.endswith(NO_DECAY_SUFFIXES):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    partition = ParameterPartition(tuple(decay), tuple(no_decay))
    partition.assert_complete(list(module.parameters()) if only is None else only)
    return partition


def adamw_groups(
    module: nn.Module,
    lr: float,
    use_decay_groups: bool,
    only: list[nn.Parameter] | None = None,
) -> list[dict]:
    parameters = list(module.parameters()) if only is None else list(only)
    if not use_decay_groups:
        return [{"params": parameters, "lr": lr}]
    return partition_parameters(module, only=only).optimizer_groups(lr)


class GeneratorTrainer:
    def __init__(
        self,
        generator: MotionGeneratorModule,
        tokenizer: TokenizerModule,
        cfg: TrainingConfig,
        downsample: int,
        text_encoder: CLIPTextEncoder | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.generator = generator
        self.tokenizer = tokenizer.eval().requires_grad_(False)
        self.text_encoder = text_encoder
        self.cfg = cfg
        self.downsample = downsample

        groups: list[dict] = []
        self._clip_params = list(generator.parameters())
        groups.extend(adamw_groups(generator, cfg.lr, cfg.decay_groups))

        if text_encoder is not None:
            encoder_trainable = [p for p in text_encoder.parameters() if p.requires_grad]
            if encoder_trainable:
                groups.extend(
                    adamw_groups(
                        text_encoder, cfg.text_encoder_lr, cfg.decay_groups, only=encoder_trainable
                    )
                )
                self._clip_params += encoder_trainable

        self.weighter: UncertaintyWeighter | None = None
        if cfg.loss_weighting == LossWeighting.UNCERTAINTY:
            self.weighter = UncertaintyWeighter(cfg.loss_weights.active_terms()).to(
                next(generator.parameters()).device
            )
            groups.append({"params": list(self.weighter.parameters()), "weight_decay": 0.0})
            self._clip_params += list(self.weighter.parameters())

        self._mean = self._std = self._skeleton = None
        if cfg.loss_weights.fk_enabled:
            if mean is None or std is None:
                raise ValueError(
                    "FK-consistency loss needs mean/std (pass them to GeneratorTrainer)"
                )
            device = next(generator.parameters()).device
            self._mean = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(device)
            self._std = torch.from_numpy(np.asarray(std, dtype=np.float32)).to(device)
            self._skeleton = Skeleton(
                torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, str(device)
            )

        self.opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.ema = Ema(generator, cfg.ema_decay)
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._accum_count = 0

    def build_scheduler(self, total_steps: int) -> None:
        warmup = max(self.cfg.warmup_steps, 1)
        floor = self.cfg.lr_min_ratio

        def lr_factor(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(total_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return floor + (1.0 - floor) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_factor)

    def encode(self, texts: list[str]) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("trainer has no text_encoder; pass a precomputed text_emb instead")
        return self.text_encoder(texts)

    def drop_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        if self.cfg.cfg_dropout <= 0:
            return text_emb
        keep_shape = (text_emb.size(0),) + (1,) * (text_emb.dim() - 1)
        keep = (torch.rand(keep_shape, device=text_emb.device) >= self.cfg.cfg_dropout).float()
        return text_emb * keep

    def _append_end_targets(
        self, target_tokens: torch.Tensor, lengths: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, t_tokens, _ = target_tokens.shape
        device = target_tokens.device
        if lengths is None:
            token_lengths = torch.full((batch,), t_tokens, device=device, dtype=torch.long)
        else:
            token_lengths = (lengths // self.downsample).clamp(min=1, max=t_tokens)
        pad = torch.full_like(target_tokens[:, :1], self.generator.end_id)
        target_tokens = torch.cat([target_tokens, pad], dim=1)
        target_tokens[torch.arange(batch, device=device), token_lengths] = self.generator.end_id
        return target_tokens, token_lengths + 1

    def train_step(
        self,
        gt_motion: torch.Tensor,
        text_emb: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        usable = (gt_motion.size(1) // self.downsample) * self.downsample
        gt_motion = gt_motion[:, :usable]
        if lengths is not None:
            lengths = lengths.clamp(max=usable)

        with torch.no_grad():
            target_tokens = self.tokenizer.encode(gt_motion)

        token_lengths = None
        if self.generator.end_id is not None:
            target_tokens, token_lengths = self._append_end_targets(target_tokens, lengths)

        input_tokens = target_tokens
        if self.cfg.pkeep < 1.0:
            corrupt = torch.rand(target_tokens.shape, device=target_tokens.device) >= self.cfg.pkeep
            random_tokens = torch.randint_like(target_tokens, self.generator.cfg.codebook_size)
            input_tokens = torch.where(corrupt, random_tokens, target_tokens)

        amp_on = self.cfg.amp == Amp.BF16 and gt_motion.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            logits = self.generator(input_tokens, self.drop_text(text_emb))
            total, parts = generator_loss(
                logits,
                target_tokens,
                gt_motion,
                self.tokenizer,
                self.cfg.loss_weights,
                self.downsample,
                lengths,
                token_lengths=token_lengths,
                has_end=self.generator.end_id is not None,
                mean=self._mean,
                std=self._std,
                skeleton=self._skeleton,
                weighter=self.weighter,
            )

        accum = max(1, self.cfg.grad_accum)
        if self._accum_count == 0:
            self.opt.zero_grad()
        (total / accum).backward()
        self._accum_count += 1
        if self._accum_count >= accum:
            self._accum_count = 0
            nn.utils.clip_grad_norm_(self._clip_params, self.cfg.grad_clip)
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.ema.update(self.generator)

        return parts

    def train_epoch(self, loader, device: str, heartbeat: Path | None = None) -> dict[str, float]:
        self.generator.train()
        if self.text_encoder is not None:
            self.text_encoder.train()

        totals: dict[str, torch.Tensor] = {}
        steps = 0
        for batch in loader:
            text_emb = self.encode(list(batch.captions))
            parts = self.train_step(batch.features.to(device), text_emb, batch.lengths.to(device))
            for key, value in parts.items():
                running = totals.get(key)
                totals[key] = value.double() if running is None else running + value.double()
            steps += 1
            if heartbeat is not None and steps % 100 == 0:
                heartbeat.write_text(str(steps), encoding="utf-8")
        return {key: value.item() / steps for key, value in totals.items()}

    @contextmanager
    def ema_weights(self):
        self.ema.copy_to(self.generator)
        try:
            yield self.generator
        finally:
            self.ema.restore(self.generator)

    def resume_state(self, epoch: int, best_fid: float) -> dict:
        return {
            "generator": self.generator.state_dict(),
            "text_encoder": self.text_encoder.state_dict() if self.text_encoder else None,
            "optimizer": self.opt.state_dict(),
            "ema": self.ema.shadow,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "best_fid": best_fid,
        }

    def load_resume_state(self, state: dict, device: str) -> tuple[int, float]:
        self.generator.load_state_dict(state["generator"])
        if self.text_encoder is not None and state.get("text_encoder") is not None:
            self.text_encoder.load_state_dict(state["text_encoder"])
        self.opt.load_state_dict(state["optimizer"])
        self.ema.shadow = {key: value.to(device) for key, value in state["ema"].items()}
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        return state["epoch"] + 1, state["best_fid"]


def train_generator(
    trainer: GeneratorTrainer,
    loader,
    request: GeneratorTrainingRequest,
    device: str,
    evaluate: Evaluator,
    save_best: CheckpointSink,
    resume_path: Path,
    run_dir: Path | None = None,
    on_metrics: MetricSink = lambda metrics: None,
) -> float:
    best_fid = float("inf")
    start_epoch = 0
    heartbeat = None if run_dir is None else Path(run_dir) / "heartbeat"

    if request.resume and resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu")
        start_epoch, best_fid = trainer.load_resume_state(state, device)
        del state
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"resumed from {resume_path} at epoch {start_epoch} (best FID {best_fid:.4f})")

    for epoch in range(start_epoch, request.epochs):
        loader.dataset.set_epoch(epoch)
        means = trainer.train_epoch(loader, device, heartbeat=heartbeat)
        on_metrics({"epoch": epoch + 1, **means})
        print(
            f"epoch {epoch + 1:3d}  ce {means['ce']:.4f}  ric {means['ric']:.4f}  "
            f"rot6d {means['rot6d']:.4f}  foot {means['foot']:.4f}  total {means['total']:.4f}"
        )

        due = (epoch + 1) % request.eval_every == 0 or epoch + 1 == request.epochs
        if due:
            with trainer.ema_weights():
                metrics = evaluate(request.eval_split)
            on_metrics({"epoch": epoch + 1, "split": request.eval_split.value, **metrics})
            print(
                f"  [gen-eval:{request.eval_split.value}] clips {metrics['clips']}  "
                f"FID {metrics['fid']:.4f}  R@1 {metrics['r_top1']:.3f}  "
                f"R@3 {metrics['r_top3']:.3f}  MM {metrics['mm_dist']:.3f}  "
                f"Div {metrics['diversity']:.3f}"
            )
            if metrics["fid"] < best_fid:
                best_fid = metrics["fid"]
                with trainer.ema_weights():
                    save_best(epoch + 1, best_fid)

        torch.save(trainer.resume_state(epoch, best_fid), resume_path)

        if device == "cuda":
            torch.cuda.empty_cache()

    return best_fid


MIN_TOKEN_ACCURACY = 0.99
MAX_CE = 0.1
MAX_TOTAL_RATIO = 0.05


@torch.no_grad()
def teacher_forced_accuracy(
    generator: MotionGeneratorModule,
    tokenizer: TokenizerModule,
    downsample: int,
    motion: torch.Tensor,
    text_embedding: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[float, float]:
    usable = (motion.size(1) // downsample) * downsample
    motion = motion[:, :usable]
    target = tokenizer.encode(motion)
    token_lengths = (lengths.clamp(max=usable) // downsample).clamp(min=1, max=target.size(1))
    logits = generator(target, text_embedding)
    valid = torch.arange(target.size(1), device=target.device)[None, :] < token_lengths[:, None]
    valid = valid[..., None].expand_as(target)
    predictions = logits.argmax(-1)
    accuracy = (predictions[valid] == target[valid]).float().mean()
    ce = nn.functional.cross_entropy(logits[valid], target[valid])
    return float(accuracy), float(ce)


def overfit_single_batch(
    trainer: GeneratorTrainer,
    motion: torch.Tensor,
    lengths: torch.Tensor,
    captions: list[str],
    steps: int,
) -> dict[str, float]:
    generator = trainer.generator
    text_encoder = trainer.text_encoder
    first_total = None
    parts: dict[str, float] = {}
    step = 0

    for step in range(steps):
        text_embedding = trainer.encode(captions)
        parts = {
            name: float(value)
            for name, value in trainer.train_step(motion, text_embedding, lengths).items()
        }
        first_total = parts["total"] if first_total is None else first_total
        if step % 25 and step + 1 != steps:
            continue
        generator.eval()
        text_encoder.eval()
        accuracy, ce = teacher_forced_accuracy(
            generator,
            trainer.tokenizer,
            trainer.downsample,
            motion,
            text_encoder(captions),
            lengths,
        )
        generator.train()
        text_encoder.train()
        print(
            f"step {step + 1:4d} total {parts['total']:.5f} ce {ce:.5f} token_accuracy {accuracy:.4f}"
        )
        if (
            accuracy >= MIN_TOKEN_ACCURACY
            and ce <= MAX_CE
            and parts["total"] <= MAX_TOTAL_RATIO * first_total
        ):
            break

    generator.eval()
    text_encoder.eval()
    accuracy, ce = teacher_forced_accuracy(
        generator, trainer.tokenizer, trainer.downsample, motion, text_encoder(captions), lengths
    )
    if any(parameter.grad is not None for parameter in trainer.tokenizer.parameters()):
        raise RuntimeError("frozen tokenizer received gradients during overfit gate")

    return {
        "steps": step + 1,
        "first_total": float(first_total),
        "final_total": parts["total"],
        "token_accuracy": accuracy,
        "ce": ce,
    }
