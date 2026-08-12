import math

import numpy as np
import torch
from torch import nn

from text2motion.data.hml3d import param_util
from text2motion.data.hml3d.skeleton import Skeleton
from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import TrainCfg
from text2motion.train.ema import Ema
from text2motion.train.losses import UncertaintyWeighter, active_term_names, generator_loss

NO_DECAY_SUFFIXES = ("a_log",)


def undecayed_param_ids(module: nn.Module) -> set[int]:
    ids: set[int] = set()
    for sub in module.modules():
        if isinstance(sub, (nn.Embedding, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
            ids |= {id(p) for p in sub.parameters(recurse=False)}
        elif type(sub).__name__ == "RMSNorm":
            ids |= {id(p) for p in sub.parameters(recurse=False)}
    return ids


def split_decay(module: nn.Module, only: list[nn.Parameter] | None = None) -> tuple[list, list]:
    exempt_ids = undecayed_param_ids(module)
    allowed = None if only is None else {id(p) for p in only}

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if allowed is not None and id(param) not in allowed:
            continue
        if param.ndim <= 1 or id(param) in exempt_ids or name.endswith(NO_DECAY_SUFFIXES):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


class GeneratorTrainer:
    def __init__(
        self,
        generator: MotionGenerator,
        tokenizer: ResidualFsqTokenizer,
        cfg: TrainCfg,
        text_encoder: CLIPTextEncoder | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.generator = generator
        self.tokenizer = tokenizer.eval().requires_grad_(
            False
        )  # frozen: differentiable, not updated
        self.text_encoder = text_encoder
        self.cfg = cfg

        groups: list[dict] = []
        self._clip_params = list(generator.parameters())

        if cfg.decay_groups:
            gen_decay, gen_plain = split_decay(generator)
            groups.append({"params": gen_decay, "lr": cfg.lr})
            groups.append({"params": gen_plain, "lr": cfg.lr, "weight_decay": 0.0})
        else:  # pre-2026-08 behaviour: one group, decay on everything (sec. 5.9)
            groups.append({"params": list(generator.parameters()), "lr": cfg.lr})

        if text_encoder is not None:
            enc_trainable = [p for p in text_encoder.parameters() if p.requires_grad]
            if enc_trainable:  # fully-frozen encoder adds nothing to the optimiser
                if cfg.decay_groups:
                    enc_decay, enc_plain = split_decay(text_encoder, only=enc_trainable)
                    groups.append({"params": enc_decay, "lr": cfg.text_encoder_lr})
                    groups.append(
                        {"params": enc_plain, "lr": cfg.text_encoder_lr, "weight_decay": 0.0}
                    )
                else:
                    groups.append({"params": enc_trainable, "lr": cfg.text_encoder_lr})
                self._clip_params += enc_trainable

        self.weighter: UncertaintyWeighter | None = None
        if cfg.loss_weighting == "uncertainty":
            self.weighter = UncertaintyWeighter(active_term_names(cfg)).to(
                next(generator.parameters()).device
            )
            groups.append({"params": list(self.weighter.parameters()), "weight_decay": 0.0})
            self._clip_params += list(self.weighter.parameters())

        self._fk_on = cfg.w_fk_self > 0 or cfg.w_fk_gt > 0
        self._mean = self._std = self._skeleton = None
        if self._fk_on:
            if mean is None or std is None:
                raise ValueError(
                    "FK-consistency loss needs mean/std (pass them to GeneratorTrainer)"
                )
            device = next(generator.parameters()).device  # FK runs on the model's device
            self._mean = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(device)
            self._std = torch.from_numpy(np.asarray(std, dtype=np.float32)).to(device)
            self._skeleton = Skeleton(
                torch.from_numpy(param_util.t2m_raw_offsets),
                param_util.t2m_kinematic_chain,
                str(device),
            )

        self.opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.ema = Ema(generator, cfg.ema_decay)
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._accum_count = 0  # micro-batches since the last optimizer step (cfg.grad_accum)

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
        self, target_tokens: torch.Tensor, lengths: torch.Tensor | None, downsample: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, t_tokens, _ = target_tokens.shape
        device = target_tokens.device
        if lengths is None:
            token_lengths = torch.full((batch,), t_tokens, device=device, dtype=torch.long)
        else:
            token_lengths = (lengths // downsample).clamp(min=1, max=t_tokens)
        pad = torch.full_like(target_tokens[:, :1], self.generator.end_id)
        target_tokens = torch.cat([target_tokens, pad], dim=1)
        target_tokens[torch.arange(batch, device=device), token_lengths] = self.generator.end_id
        return target_tokens, token_lengths + 1

    def train_step(
        self,
        gt_motion: torch.Tensor,
        text_emb: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, float]:
        downsample = self.tokenizer.cfg.downsample
        usable = (gt_motion.size(1) // downsample) * downsample  # encode/decode align on multiples
        gt_motion = gt_motion[:, :usable]
        if lengths is not None:
            lengths = lengths.clamp(max=usable)

        with torch.no_grad():
            target_tokens = self.tokenizer.encode(gt_motion)  # (B, T', R)

        token_lengths = None
        if self.generator.end_id is not None:
            target_tokens, token_lengths = self._append_end_targets(
                target_tokens, lengths, downsample
            )

        input_tokens = target_tokens
        if (
            self.cfg.pkeep < 1.0
        ):  # corrupt teacher-forcing INPUTS only; targets stay clean (T2M-GPT)
            corrupt = torch.rand(target_tokens.shape, device=target_tokens.device) >= self.cfg.pkeep
            random_tokens = torch.randint_like(target_tokens, self.generator.cfg.codebook_size)
            input_tokens = torch.where(corrupt, random_tokens, target_tokens)

        amp_on = self.cfg.amp == "bf16" and gt_motion.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            logits = self.generator(input_tokens, self.drop_text(text_emb))
            total, parts = generator_loss(
                logits,
                target_tokens,
                gt_motion,
                self.tokenizer,
                self.cfg,
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
