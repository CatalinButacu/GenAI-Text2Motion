"""Adapter helpers for fine-tuning TextToMotionSSM with most params frozen.

Used by the ``--mode adapter`` path in scripts/training/train_motion_ssm.py.

Two related primitives:

1. ``freeze_trunk(model, unfreeze_film=True, unfreeze_decoder=True)``
   Walks ``TextToMotionSSM`` parameter groups and disables ``requires_grad`` for
   the SSM trunk (``layers.*``), positional embeddings, and the text encoder.
   Optional knobs leave FiLM gamma/beta and the RVQ decoder head trainable.

2. ``LoRALinear`` — drop-in replacement for ``nn.Linear`` with a low-rank
   delta. The base weight is frozen; only the rank-r adapter is updated.
   Use when even FiLM + decoder is too many parameters for the available data,
   or to inject task-specific deltas into ``condition_proj``.

These are intentionally small and self-contained — no monkey-patching of the
trunk, no global state. Use ``freeze_trunk`` to set requires_grad and call
``replace_linear_with_lora(model, "condition_proj", rank=16)`` if you want a
LoRA adapter on a specific submodule.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """nn.Linear + frozen base weight + trainable rank-r delta (A @ B).

    Forward: y = x W_base^T + (x A) B    where A: (in, r), B: (r, out)
    Only A and B receive gradients; W_base is kept as a buffer-like Parameter
    with ``requires_grad=False`` so the optimizer skips it.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float | None = None, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = (alpha if alpha is not None else rank) / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.bias = None

        self.lora_a = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)  # start at zero -> identity to base

    @classmethod
    def from_linear(cls, lin: nn.Linear, rank: int = 8,
                    alpha: float | None = None) -> LoRALinear:
        mod = cls(lin.in_features, lin.out_features, rank=rank, alpha=alpha,
                  bias=lin.bias is not None)

        with torch.no_grad():
            mod.weight.copy_(lin.weight)

            if lin.bias is not None and mod.bias is not None:
                mod.bias.copy_(lin.bias)

        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        delta = (x @ self.lora_a) @ self.lora_b * self.scaling

        return base + delta


def freeze_trunk(model: nn.Module, unfreeze_film: bool = True,
                 unfreeze_decoder: bool = True,
                 unfreeze_condition_proj: bool = True) -> dict:
    """Disable requires_grad on the SSM trunk + text encoder + positional embedding.

    Optional knobs keep FiLM, the RVQ decoder head, and condition_proj trainable.

    Returns a dict with ``{trainable, total, modules_frozen, modules_trainable}``.
    """
    trainable_modules: list[str] = []
    frozen_modules: list[str] = []

    for name, param in model.named_parameters():
        # Trunk (dominant param count) = Mamba stack + pos embedding + text encoder.
        is_trunk = name.startswith(("layers.", "pos_embed.", "text_encoder."))
        is_film = name.startswith("films.")
        is_decoder = name.startswith("decoder.")
        is_condition_proj = name.startswith("condition_proj.")
        is_seed_projector = name.startswith("seed_projector.")

        train = (
            (unfreeze_film and is_film)
            or (unfreeze_decoder and is_decoder)
            or (unfreeze_condition_proj and is_condition_proj)
            or (not is_trunk and not is_film and not is_decoder
                and not is_condition_proj and not is_seed_projector)
        )
        # Always freeze trunk + text encoder + pos embed regardless of flags.

        if is_trunk:
            train = False

        param.requires_grad = bool(train)

        if train:
            trainable_modules.append(name)
        else:
            frozen_modules.append(name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(
        "frozen %d params, trainable %d params (%.1f%% trainable)",
        n_total - n_trainable, n_trainable,
        100.0 * n_trainable / max(n_total, 1),
    )

    return {
        "trainable": n_trainable,
        "total": n_total,
        "modules_trainable": trainable_modules,
        "modules_frozen": frozen_modules,
    }


def replace_linear_with_lora(parent: nn.Module, attr_path: str, rank: int = 8,
                             alpha: float | None = None) -> LoRALinear:
    """Wrap an existing nn.Linear submodule with LoRALinear in place.

    ``attr_path`` is a dotted path like "condition_proj" or "decoder.head". The
    target must currently be an ``nn.Linear``.
    """
    parts = attr_path.split(".")
    holder = parent

    for p in parts[:-1]:
        holder = getattr(holder, p)
    target = getattr(holder, parts[-1])

    if not isinstance(target, nn.Linear):
        raise TypeError(f"{attr_path} is {type(target).__name__}, not nn.Linear")

    new_mod = LoRALinear.from_linear(target, rank=rank, alpha=alpha)
    setattr(holder, parts[-1], new_mod)
    log.info("%s -> LoRALinear(rank=%d)", attr_path, rank)

    return new_mod
