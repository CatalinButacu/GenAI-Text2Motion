from __future__ import annotations

import torch
from torch import nn


############################################################
#       ExponentialMovingAverage
############################################################
class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            shadow.mul_(self.decay).add_(param, alpha=1 - self.decay)

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing = sorted(self.shadow.keys() - state.keys())
        unexpected = sorted(state.keys() - self.shadow.keys())
        if missing or unexpected:
            raise KeyError(f"EMA state keys differ: missing {missing}, unexpected {unexpected}")
        for name, shadow in self.shadow.items():
            value = state[name]
            if value.shape != shadow.shape:
                raise ValueError(f"EMA {name}: shape {tuple(value.shape)} != {tuple(shadow.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"EMA {name}: the checkpoint value is not finite")
            shadow.copy_(value)

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
