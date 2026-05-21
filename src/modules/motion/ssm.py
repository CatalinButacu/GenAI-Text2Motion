from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.shared.constants import SSM_D_MODEL, SSM_D_STATE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SSMConfig:
    d_model: int = SSM_D_MODEL
    d_state: int = SSM_D_STATE
    d_conv: int = 4
    expand: int = 2
    dt_rank: str | int = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    gradient_checkpointing: bool = False

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def ssm_scan(
    a_bar_t: torch.Tensor,
    bx_t: torch.Tensor,
    c_sel: torch.Tensor,
    d_skip: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    # NOTE: @torch.jit.script intentionally removed -- it conflicts with
    # torch.utils.checkpoint during the backward recompute pass.
    B, T, d_inner = x.shape
    d_state = a_bar_t.shape[3]
    h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
    out = torch.zeros(B, T, d_inner, device=x.device, dtype=x.dtype)

    for t in range(T):
        h = a_bar_t[:, t] * h + bx_t[:, t]
        out[:, t] = (h * c_sel[:, t, :].unsqueeze(1)).sum(-1) + d_skip * x[:, t]

    return out


class MambaLayer(nn.Module):
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_state = config.d_state
        d_inner = config.d_inner
        self.dt_rank = config.dt_rank if isinstance(config.dt_rank, int) else d_model // 16
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            config.d_conv,
            padding=config.d_conv - 1,
            groups=d_inner,
        )
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.init_dt()

    def init_dt(self):
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.config.d_inner)
            * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        ).clamp(min=1e-4)
        self.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :length].transpose(1, 2))

        if self.config.gradient_checkpointing:
            y = checkpoint(self.ssm_forward, x, use_reentrant=False)
        else:
            y = self.ssm_forward(x)

        return self.out_proj(y * F.silu(z))  # type: ignore[operator]

    def ssm_forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        d_state = self.config.d_state
        x_dbl = self.x_proj(x)
        dt, b_sel, c_sel = x_dbl.split([self.dt_rank, d_state, d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        a_diag = -torch.exp(self.A_log)
        dt3d = dt.unsqueeze(-1)
        a_bar_t = torch.exp(dt3d * a_diag)
        bx_t = dt3d * b_sel.unsqueeze(2) * x.unsqueeze(-1)
        y = ssm_scan(a_bar_t, bx_t, c_sel, self.D, x)

        if length == 0:
            return torch.zeros_like(x)

        return y

    def step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        conv_buf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-frame autoregressive step.

        Args:
            x_t:      (B, d_model)            -- input for timestep t.
            h:        (B, d_inner, d_state)   -- recurrent SSM state from t-1.
            conv_buf: (B, d_inner, d_conv-1)  -- rolling conv input history.
                      Pass None on the first frame; a zero buffer is created.

        Returns:
            out:      (B, d_model)            -- output for timestep t.
            h:        (B, d_inner, d_state)   -- updated state.
            conv_buf: (B, d_inner, d_conv-1)  -- updated history for next call.
        """
        x_t3 = x_t.unsqueeze(1)  # (B, 1, d_model)
        xz = self.in_proj(x_t3)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, 1, d_inner)

        if h.shape[1] != self.config.d_inner or h.shape[2] != self.config.d_state:
            raise ValueError(
                f"step(): h shape {tuple(h.shape)} does not match expected "
                f"(B, {self.config.d_inner}, {self.config.d_state})"
            )

        # Maintain a rolling buffer of the last d_conv-1 frames so conv1d sees
        # the same context it would in the parallel forward pass.
        x_inner_t = x_inner.squeeze(1)  # (B, d_inner)
        d_conv_m1 = self.config.d_conv - 1

        if conv_buf is None:
            conv_buf = torch.zeros(
                x_inner_t.shape[0],
                x_inner_t.shape[1],
                d_conv_m1,
                device=x_t.device,
                dtype=x_t.dtype,
            )
        # Concatenate history + current frame -> (B, d_inner, d_conv)
        conv_input = torch.cat([conv_buf, x_inner_t.unsqueeze(-1)], dim=-1)
        # Slide buffer forward: drop oldest frame, keep last d_conv-1 frames.
        conv_buf = conv_input[:, :, 1:].detach()
        # Apply depthwise conv (no extra padding; input is exactly d_conv long).
        x_inner_conv = F.silu(
            F.conv1d(
                conv_input,
                self.conv1d.weight,
                self.conv1d.bias,
                groups=self.conv1d.groups,
            )[:, :, 0]
        )  # (B, d_inner)

        x_dbl = self.x_proj(x_inner_conv)  # (B, dt_rank + 2*d_state)
        dt, b_sel, c_sel = x_dbl.split(
            [self.dt_rank, self.config.d_state, self.config.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, d_inner)
        a_diag = -torch.exp(self.A_log)  # (d_state,)

        # ZOH discretisation
        a_bar = torch.exp(dt.unsqueeze(-1) * a_diag)  # (B, d_inner, d_state)
        bx = (
            dt.unsqueeze(-1) * b_sel.unsqueeze(1) * x_inner_conv.unsqueeze(-1)
        )  # (B, d_inner, d_state)

        h = a_bar * h + bx  # (B, d_inner, d_state)
        y = (h * c_sel.unsqueeze(1)).sum(-1) + self.D * x_inner_conv  # (B, d_inner)

        out = self.out_proj(y * F.silu(z.squeeze(1)))  # (B, d_model)

        return out, h, conv_buf


class BiMambaLayer(nn.Module):
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.fwd = MambaLayer(config)
        self.bwd = MambaLayer(config)
        self.merge = nn.Linear(config.d_model * 2, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_out = self.fwd(x)
        bwd_out = self.bwd(x.flip(1)).flip(1)

        return self.merge(torch.cat([fwd_out, bwd_out], dim=-1))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

LAYER_BUILDERS = {
    "mamba": lambda **kw: MambaLayer(SSMConfig(**kw)),
}


def create_ssm_layer(layer_type: str = "mamba", **kwargs):
    if (builder := LAYER_BUILDERS.get(layer_type)) is None:
        raise ValueError(f"Unknown layer type: {layer_type}")

    return builder(**kwargs)


def get_ssm_info() -> dict:
    return {
        "torch_available": True,
        "layers": ["mamba", "bimamba"],
        "references": {
            "S4": "https://arxiv.org/abs/2111.00396",
            "Mamba": "https://arxiv.org/abs/2312.00752",
            "Motion Mamba": "https://arxiv.org/abs/2403.07487",
            "HiPPO": "https://arxiv.org/abs/2008.07669",
        },
        "novel_contribution": "MotionSSM --Pure Mamba Motion Synthesis",
    }


__all__ = [
    "SSMConfig",
    "MambaLayer",
    "BiMambaLayer",
    "create_ssm_layer",
    "get_ssm_info",
]
