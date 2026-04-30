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
    dModel: int = SSM_D_MODEL
    dState: int = SSM_D_STATE
    dConv: int = 4
    expand: int = 2
    dtRank: str | int = "auto"
    dtMin: float = 0.001
    dtMax: float = 0.1
    dtInit: str = "random"
    gradientCheckpointing: bool = False

    def __post_init__(self):
        self.d_inner = self.expand * self.dModel

        if self.dtRank == "auto":
            self.dtRank = math.ceil(self.dModel / 16)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def ssmScan(
    aBarT: torch.Tensor,
    bxT: torch.Tensor,
    cSel: torch.Tensor,
    dSkip: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    # NOTE: @torch.jit.script intentionally removed -- it conflicts with
    # torch.utils.checkpoint during the backward recompute pass.
    B, T, d_inner = x.shape
    dState = aBarT.shape[3]
    h = torch.zeros(B, d_inner, dState, device=x.device, dtype=x.dtype)
    out = torch.zeros(B, T, d_inner, device=x.device, dtype=x.dtype)

    for t in range(T):
        h = aBarT[:, t] * h + bxT[:, t]
        out[:, t] = (h * cSel[:, t, :].unsqueeze(1)).sum(-1) + dSkip * x[:, t]

    return out


class MambaLayer(nn.Module):
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        dModel = config.dModel
        dState = config.dState
        dInner = config.d_inner
        self.dtRank = config.dtRank if isinstance(config.dtRank, int) else dModel // 16
        self.in_proj = nn.Linear(dModel, dInner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            dInner,
            dInner,
            config.dConv,
            padding=config.dConv - 1,
            groups=dInner,
        )
        aInit = torch.arange(1, dState + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(aInit))
        self.D = nn.Parameter(torch.ones(dInner))
        self.x_proj = nn.Linear(dInner, self.dtRank + dState * 2, bias=False)
        self.dt_proj = nn.Linear(self.dtRank, dInner, bias=True)
        self.out_proj = nn.Linear(dInner, dModel, bias=False)
        self.initDt()

    def initDt(self):
        std = self.dtRank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.config.d_inner)
            * (math.log(self.config.dtMax) - math.log(self.config.dtMin))
            + math.log(self.config.dtMin)
        ).clamp(min=1e-4)
        self.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :length].transpose(1, 2))

        if self.config.gradientCheckpointing:
            y = checkpoint(self.ssmForward, x, use_reentrant=False)
        else:
            y = self.ssmForward(x)

        return self.out_proj(y * F.silu(z))  # type: ignore[operator]

    def ssmForward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        dState = self.config.dState
        xDbl = self.x_proj(x)
        dt, b_sel, c_sel = xDbl.split([self.dtRank, dState, dState], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        aDiag = -torch.exp(self.A_log)
        dt3d = dt.unsqueeze(-1)
        aBarT = torch.exp(dt3d * aDiag)
        bxT = dt3d * b_sel.unsqueeze(2) * x.unsqueeze(-1)
        y = ssmScan(aBarT, bxT, c_sel, self.D, x)

        if length == 0:
            return torch.zeros_like(x)

        return y

    def step(
        self,
        xT: torch.Tensor,
        h: torch.Tensor,
        convBuf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-frame autoregressive step.

        Args:
            xT:      (B, d_model)            -- input for timestep t.
            h:        (B, d_inner, d_state)   -- recurrent SSM state from t-1.
            conv_buf: (B, d_inner, d_conv-1)  -- rolling conv input history.
                      Pass None on the first frame; a zero buffer is created.

        Returns:
            out:      (B, d_model)            -- output for timestep t.
            h:        (B, d_inner, d_state)   -- updated state.
            conv_buf: (B, d_inner, d_conv-1)  -- updated history for next call.
        """
        xT3 = xT.unsqueeze(1)  # (B, 1, d_model)
        xz = self.in_proj(xT3)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, 1, d_inner)

        if h.shape[1] != self.config.d_inner or h.shape[2] != self.config.dState:
            raise ValueError(
                f"step(): h shape {tuple(h.shape)} does not match expected "
                f"(B, {self.config.d_inner}, {self.config.dState})"
            )

        # Maintain a rolling buffer of the last d_conv-1 frames so conv1d sees
        # the same context it would in the parallel forward pass.
        xInnerT = x_inner.squeeze(1)  # (B, d_inner)
        dConvM1 = self.config.dConv - 1

        if convBuf is None:
            convBuf = torch.zeros(
                xInnerT.shape[0],
                xInnerT.shape[1],
                dConvM1,
                device=xT.device,
                dtype=xT.dtype,
            )
        # Concatenate history + current frame -> (B, d_inner, d_conv)
        convInput = torch.cat([convBuf, xInnerT.unsqueeze(-1)], dim=-1)
        # Slide buffer forward: drop oldest frame, keep last d_conv-1 frames.
        convBuf = convInput[:, :, 1:].detach()
        # Apply depthwise conv (no extra padding; input is exactly d_conv long).
        xInnerConv = F.silu(
            F.conv1d(
                convInput,
                self.conv1d.weight,
                self.conv1d.bias,
                groups=self.conv1d.groups,
            )[:, :, 0]
        )  # (B, d_inner)

        xDbl = self.x_proj(xInnerConv)  # (B, dt_rank + 2*d_state)
        dt, b_sel, c_sel = xDbl.split(
            [self.dtRank, self.config.dState, self.config.dState], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, d_inner)
        aDiag = -torch.exp(self.A_log)  # (d_state,)

        # ZOH discretisation
        aBar = torch.exp(dt.unsqueeze(-1) * aDiag)  # (B, d_inner, d_state)
        bx = (
            dt.unsqueeze(-1) * b_sel.unsqueeze(1) * xInnerConv.unsqueeze(-1)
        )  # (B, d_inner, d_state)

        h = aBar * h + bx  # (B, d_inner, d_state)
        y = (h * c_sel.unsqueeze(1)).sum(-1) + self.D * xInnerConv  # (B, d_inner)

        out = self.out_proj(y * F.silu(z.squeeze(1)))  # (B, d_model)

        return out, h, convBuf


class BiMambaLayer(nn.Module):
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.fwd = MambaLayer(config)
        self.bwd = MambaLayer(config)
        self.merge = nn.Linear(config.dModel * 2, config.dModel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwdOut = self.fwd(x)
        bwdOut = self.bwd(x.flip(1)).flip(1)

        return self.merge(torch.cat([fwdOut, bwdOut], dim=-1))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

LAYER_BUILDERS = {
    "mamba": lambda **kw: MambaLayer(SSMConfig(**kw)),
}


def createSsmLayer(layerType: str = "mamba", **kwargs):
    if (builder := LAYER_BUILDERS.get(layerType)) is None:
        raise ValueError(f"Unknown layer type: {layerType}")

    return builder(**kwargs)


def getSsmInfo() -> dict:
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
    "createSsmLayer",
    "getSsmInfo",
]
