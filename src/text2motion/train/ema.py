"""Exponential moving average of model weights (decay 0.999). The prior run kept noisy last-step
weights; MDM/MoMask both evaluate an EMA copy. Use `store`/`copy_to`/`restore` around evaluation."""

import torch
from torch import nn


class Ema:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """Swap EMA weights into the model, stashing the live weights for `restore`."""
        self.backup = {
            n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow
        }

        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

        self.backup = {}
