from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from text2motion.generation.contracts import (
    Backbone,
    GeneratorConfig,
)
from text2motion.generation.contracts import BackboneChoice as BackboneChoice


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


def parallel_scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    length = a.size(1)
    shift = 1
    while shift < length:
        pad = (0, 0, 0, 0, shift, 0)
        a_prev = F.pad(a[:, :-shift], pad, value=1.0)
        x_prev = F.pad(x[:, :-shift], pad, value=0.0)
        x = x + a * x_prev
        a = a * a_prev
        shift *= 2
    return x


class MambaMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
        use_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        self.use_kernel = use_kernel
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d_skip = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def init_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ssm = torch.zeros(batch, self.d_inner, self.d_state, device=device)
        conv = torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device)
        return ssm, conv

    def negative_a(self) -> torch.Tensor:
        if torch.is_grad_enabled():
            return -torch.exp(self.a_log)
        version = self.a_log._version
        cached = getattr(self, "_negative_a", None)
        if cached is None or cached[0] != version:
            cached = (version, -torch.exp(self.a_log))
            self._negative_a = cached
        return cached[1]

    def step(
        self, u_t: torch.Tensor, ssm_state: torch.Tensor, conv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, z = self.in_proj(u_t).chunk(2, dim=-1)
        conv_in = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)
        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias
        if bias is None:
            raise ValueError("the Mamba convolution needs a bias")
        x_conv = (conv_in * weight).sum(-1) + bias
        new_conv = conv_in[..., 1:]
        x_c = F.silu(x_conv)

        a = self.negative_a()
        dt, b, c = torch.split(self.x_proj(x_c), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        da = torch.exp(dt.unsqueeze(-1) * a)
        dbx = (dt * x_c).unsqueeze(-1) * b.unsqueeze(1)
        new_ssm = da * ssm_state
        new_ssm += dbx
        y = (new_ssm * c.unsqueeze(1)).sum(-1) + self.d_skip * x_c
        out = self.out_proj(y * F.silu(z))
        return out, new_ssm, new_conv

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x, z = self.in_proj(seq).chunk(2, dim=-1)

        x_pad = F.pad(x.transpose(1, 2), (self.d_conv - 1, 0))
        x_c = F.silu(self.conv1d(x_pad).transpose(1, 2))

        a = -torch.exp(self.a_log)
        dt_raw, b, c = torch.split(
            self.x_proj(x_c), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        if self.use_kernel:
            y = selective_scan_function()(
                x_c.transpose(1, 2).contiguous(),
                self.dt_proj(dt_raw).transpose(1, 2).contiguous(),
                a,
                b.transpose(1, 2).contiguous(),
                c.transpose(1, 2).contiguous(),
                self.d_skip,
                delta_softplus=True,
            ).transpose(1, 2)
            return self.out_proj(y * F.silu(z))

        dt = F.softplus(self.dt_proj(dt_raw))
        da = torch.exp(dt.unsqueeze(-1) * a)
        dbx = (dt * x_c).unsqueeze(-1) * b.unsqueeze(2)

        ssm = parallel_scan(da, dbx)
        y = (ssm * c.unsqueeze(2)).sum(-1) + self.d_skip * x_c
        return self.out_proj(y * F.silu(z))


class MambaBlock(nn.Module):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MambaMixer(
            cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand, cfg.dt_rank, cfg.use_kernel
        )
        self.drop = nn.Dropout(cfg.dropout)

    def init_state(self, batch: int, device: torch.device):
        return self.mixer.init_state(batch, device)

    def step(self, x_t: torch.Tensor, state):
        out, ssm, conv = self.mixer.step(self.norm(x_t), state[0], state[1])
        return x_t + out, (ssm, conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mixer(self.norm(x)))


class SequenceBackbone(nn.Module, ABC):
    @property
    @abstractmethod
    def max_positions(self) -> int | None: ...

    @abstractmethod
    def init_state(self, batch: int, device: torch.device) -> list: ...

    @abstractmethod
    def step(self, x_t: torch.Tensor, state: list) -> tuple[torch.Tensor, list]: ...


class MambaBackbone(SequenceBackbone):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.checkpoint_blocks = not cfg.use_kernel

    def typed_blocks(self) -> list[MambaBlock]:
        return cast("list[MambaBlock]", list(self.blocks))

    @property
    def max_positions(self) -> int | None:
        return None

    def init_state(self, batch: int, device: torch.device) -> list:
        return [b.init_state(batch, device) for b in self.typed_blocks()]

    def step(self, x_t: torch.Tensor, state: list) -> tuple[torch.Tensor, list]:
        for i, block in enumerate(self.typed_blocks()):
            x_t, state[i] = block.step(x_t, state[i])

        return self.norm_f(x_t), state

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.training and seq.requires_grad and self.checkpoint_blocks:
                seq = cast(torch.Tensor, checkpoint(block, seq, use_reentrant=False))
            else:
                seq = block(seq)

        return self.norm_f(seq)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.attn_drop = cfg.dropout

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (self.split_heads(t) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_drop if self.training else 0.0
        )
        a = a.transpose(1, 2).reshape(x.shape)
        x = x + self.drop(self.proj(a))
        return x + self.drop(self.mlp(self.norm2(x)))

    def prefill(self, x: torch.Tensor):
        q, k, v = (self.split_heads(t) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(x.shape)
        x = x + self.proj(a)
        x = x + self.mlp(self.norm2(x))
        return x, (k, v)

    def step(self, x_t: torch.Tensor, cache):
        q, k, v = (self.split_heads(t.unsqueeze(1)) for t in self.qkv(self.norm1(x_t)).chunk(3, -1))

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        a = a.transpose(1, 2).reshape(x_t.shape)
        x_t = x_t + self.proj(a)
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, (k, v)


class TransformerBackbone(SequenceBackbone):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.pos = nn.Embedding(cfg.max_seq_len + 1, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

    def typed_blocks(self) -> list[TransformerBlock]:
        return cast("list[TransformerBlock]", list(self.blocks))

    @property
    def max_positions(self) -> int | None:
        return self.pos.num_embeddings

    def init_state(self, batch: int, device: torch.device) -> list:
        return [None for _ in self.blocks]

    def prefill(self, seq: torch.Tensor) -> tuple[torch.Tensor, list]:
        positions = torch.arange(seq.size(1), device=seq.device)
        seq = seq + self.pos(positions).unsqueeze(0)
        state = []
        for block in self.typed_blocks():
            seq, cache = block.prefill(seq)
            state.append(cache)
        return self.norm_f(seq[:, -1]), state

    def step(self, x_t: torch.Tensor, state: list) -> tuple[torch.Tensor, list]:
        pos = 0 if state[0] is None else state[0][0].size(2)
        x_t = x_t + self.pos(torch.full((x_t.size(0),), pos, dtype=torch.long, device=x_t.device))

        for i, block in enumerate(self.typed_blocks()):
            x_t, state[i] = block.step(x_t, state[i])

        return self.norm_f(x_t), state

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(seq.size(1), device=seq.device)
        seq = seq + self.pos(pos).unsqueeze(0)

        for block in self.blocks:
            seq = block(seq)

        return self.norm_f(seq)


_BACKBONE_FACTORIES = {
    Backbone.MAMBA: MambaBackbone,
    Backbone.TRANSFORMER: TransformerBackbone,
}


def instantiate_backbone(cfg: GeneratorConfig) -> SequenceBackbone:
    build = _BACKBONE_FACTORIES.get(Backbone(cfg.backbone))
    if build is None:
        raise ValueError(
            f"unknown backbone {cfg.backbone!r} (expected one of {[b.value for b in Backbone]})"
        )
    return build(cfg)


INTERFACE_MODULE = "mamba_ssm.ops.selective_scan_interface"
TRITON_MODULES = ("mamba_ssm.ops.triton", "mamba_ssm.ops.triton.layer_norm")


def __register_package_without_triton() -> None:
    package_spec = importlib.util.find_spec("mamba_ssm")
    if package_spec is None or package_spec.origin is None:
        raise ModuleNotFoundError(
            "use_kernel needs the mamba_ssm package; build it from third_party/ "
            "(see doc/notes/implementation-difficulties.md)"
        )
    package_dir = Path(package_spec.origin).parent
    for name, folder in (("mamba_ssm", package_dir), ("mamba_ssm.ops", package_dir / "ops")):
        package_stub = types.ModuleType(name)
        package_stub.__path__ = [str(folder)]
        sys.modules[name] = package_stub
    for name in TRITON_MODULES:
        triton_stub = types.ModuleType(name)
        triton_stub.__path__ = []
        sys.modules[name] = triton_stub
    sys.modules[TRITON_MODULES[1]].__dict__["_layer_norm_fwd"] = None


def selective_scan_function() -> Callable:
    if INTERFACE_MODULE not in sys.modules and importlib.util.find_spec("triton") is None:
        __register_package_without_triton()
    return importlib.import_module(INTERFACE_MODULE).selective_scan_fn
