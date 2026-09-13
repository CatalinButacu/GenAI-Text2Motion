from __future__ import annotations

import torch
from torch import nn


class ExponentialMovingAverage:
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
