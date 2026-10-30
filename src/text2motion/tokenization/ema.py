from __future__ import annotations

import torch
from torch import nn
from torch.optim.swa_utils import get_ema_multi_avg_fn


############################################################
#       ExponentialMovingAverage
############################################################
class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float, average_buffers: bool = False) -> None:
        self.decay = decay
        self.average_buffers = average_buffers
        self.__average = get_ema_multi_avg_fn(decay)
        trained = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        floating_buffers = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
            if average_buffers and buffer.is_floating_point()
        }
        self.shadow = trained | floating_buffers
        self.backup: dict[str, torch.Tensor] = {}

    def named_tensors(self, model: nn.Module) -> dict[str, torch.Tensor]:
        parameters: dict[str, torch.Tensor] = dict(model.named_parameters())
        if not self.average_buffers:
            return parameters
        return parameters | dict(model.named_buffers())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        tensors = self.named_tensors(model)
        current: list[torch.Tensor] = [tensors[name] for name in self.shadow]
        self.__average(list(self.shadow.values()), current, None)

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

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        if self.backup:
            raise RuntimeError(
                "copy_to was called again before restore; the trained weights would be lost"
            )
        tensors = self.named_tensors(model)
        self.backup = {name: tensors[name].detach().clone() for name in self.shadow}
        for name, shadow in self.shadow.items():
            tensors[name].copy_(shadow)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            raise RuntimeError(
                "restore was called without copy_to; there are no weights to restore"
            )
        tensors = self.named_tensors(model)
        for name, saved in self.backup.items():
            tensors[name].copy_(saved)
        self.backup = {}
