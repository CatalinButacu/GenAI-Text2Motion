from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.generation.contracts import (
    GeneratorConfig,
)
from text2motion.generation.model.layers import (
    RMSNorm,
    SequenceBackbone,
    init_reference_embedding,
    init_residual_output,
    zero_linear_bias,
)


############################################################
#       TransformerBlock
############################################################
class TransformerBlock(nn.Module):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        init_residual_output(self.proj, 2 * cfg.n_layers)
        mlp_in = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        mlp_out = nn.Linear(4 * cfg.d_model, cfg.d_model)
        zero_linear_bias(mlp_in)
        zero_linear_bias(mlp_out)
        init_residual_output(mlp_out, 2 * cfg.n_layers)
        self.mlp = nn.Sequential(mlp_in, nn.GELU(), mlp_out)

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (self.split_heads(t) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        a = a.transpose(1, 2).reshape(x.shape)
        x = x + self.proj(a)
        return x + self.mlp(self.norm2(x))

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


############################################################
#       TransformerBackbone
############################################################
class TransformerBackbone(SequenceBackbone):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.pos = nn.Embedding(cfg.max_seq_len + 1, cfg.d_model)
        init_reference_embedding(self.pos)
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
