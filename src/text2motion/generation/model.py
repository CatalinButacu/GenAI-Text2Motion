from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Backbone(StrEnum):
    TRANSFORMER = "transformer"
    MAMBA = "mamba"


class BackboneChoice(StrEnum):
    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    BOTH = "both"


@dataclass(frozen=True)
class GeneratorConfig:
    backbone: Backbone = Backbone.MAMBA
    d_model: int = 512
    n_layers: int = 8
    mamba_n_layers: int = 15
    d_text: int = 512
    num_codebooks: int = 6
    codebook_size: int = 1000
    max_seq_len: int = 96
    dropout: float = 0.1
    text_prefix_len: int = 1
    use_end_token: bool = False
    use_kernel: bool = False
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 32
    n_heads: int = 8


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


def _parallel_scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
        x_conv = (conv_in * weight).sum(-1) + self.conv1d.bias
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
        batch, length, _ = seq.shape
        x, z = self.in_proj(seq).chunk(2, dim=-1)

        x_pad = F.pad(x.transpose(1, 2), (self.d_conv - 1, 0))
        x_c = F.silu(self.conv1d(x_pad).transpose(1, 2))

        a = -torch.exp(self.a_log)
        dt_raw, b, c = torch.split(
            self.x_proj(x_c), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        if self.use_kernel:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            y = selective_scan_fn(
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

        ssm = _parallel_scan(da, dbx)
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


class MambaBackbone(nn.Module):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.checkpoint_blocks = not cfg.use_kernel

    def init_state(self, batch: int, device: torch.device) -> list:
        return [b.init_state(batch, device) for b in self.blocks]

    def step(self, x_t: torch.Tensor, state: list) -> tuple[torch.Tensor, list]:
        for i, block in enumerate(self.blocks):
            x_t, state[i] = block.step(x_t, state[i])

        return self.norm_f(x_t), state

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.training and seq.requires_grad and self.checkpoint_blocks:
                seq = checkpoint(block, seq, use_reentrant=False)
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


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.pos = nn.Embedding(cfg.max_seq_len + 1, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

    def init_state(self, batch: int, device: torch.device) -> list:
        return [None for _ in self.blocks]

    def prefill(self, seq: torch.Tensor) -> tuple[torch.Tensor, list]:
        positions = torch.arange(seq.size(1), device=seq.device)
        seq = seq + self.pos(positions).unsqueeze(0)
        state = []
        for block in self.blocks:
            seq, cache = block.prefill(seq)
            state.append(cache)
        return self.norm_f(seq[:, -1]), state

    def step(self, x_t: torch.Tensor, state: list) -> tuple[torch.Tensor, list]:
        pos = 0 if state[0] is None else state[0][0].size(2)
        x_t = x_t + self.pos(torch.full((x_t.size(0),), pos, dtype=torch.long, device=x_t.device))

        for i, block in enumerate(self.blocks):
            x_t, state[i] = block.step(x_t, state[i])

        return self.norm_f(x_t), state

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(seq.size(1), device=seq.device)
        seq = seq + self.pos(pos).unsqueeze(0)

        for block in self.blocks:
            seq = block(seq)

        return self.norm_f(seq)


class MotionGeneratorModule(nn.Module):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.end_id = cfg.codebook_size if cfg.use_end_token else None
        vocab = cfg.codebook_size + (1 if cfg.use_end_token else 0)
        self.token_emb = nn.ModuleList(
            [nn.Embedding(vocab, cfg.d_model) for _ in range(cfg.num_codebooks)]
        )
        self.text_proj = nn.Linear(cfg.d_text, cfg.d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, vocab) for _ in range(cfg.num_codebooks)]
        )
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.backbone = make_backbone(cfg)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.token_emb[0](tokens[..., 0])

        for r in range(1, self.cfg.num_codebooks):
            out = out + self.token_emb[r](tokens[..., r])

        return out

    def text_prefix(self, text_emb: torch.Tensor) -> torch.Tensor:
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)
        if text_emb.size(1) != self.cfg.text_prefix_len:
            raise ValueError(
                f"text prefix has {text_emb.size(1)} tokens, cfg.text_prefix_len is "
                f"{self.cfg.text_prefix_len} (TextEncoderCfg.prefix_len must match)"
            )
        return self.text_proj(text_emb)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(h) for head in self.heads], dim=-2)

    def forward(self, tokens: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(tokens)
        prefix = self.text_prefix(text_emb)
        seq_in = torch.cat([prefix, emb[:, :-1]], dim=1)
        h = self.backbone(self.emb_drop(seq_in))[:, prefix.size(1) - 1 :]
        return self.logits(h)

    @torch.no_grad()
    def stream(
        self,
        text_emb: torch.Tensor,
        steps: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        cfg_scale: float = 1.0,
        stop_at_end: bool = False,
        uniforms: torch.Tensor | None = None,
    ):
        device = text_emb.device
        guided = cfg_scale != 1.0
        if guided:
            text_emb = torch.cat([text_emb, torch.zeros_like(text_emb)], dim=0)
        prefix = self.text_prefix(text_emb)
        if hasattr(self.backbone, "prefill"):
            h, state = self.backbone.prefill(prefix)
        else:
            state = self.backbone.init_state(text_emb.size(0), device)
            for position in range(prefix.size(1)):
                h, state = self.backbone.step(prefix[:, position], state)

        for index in range(steps):
            logits = self.logits(h)
            if guided:
                cond, uncond = logits.chunk(2, dim=0)
                logits = uncond + cfg_scale * (cond - uncond)
            if self.end_id is not None and not stop_at_end:
                logits[..., self.end_id] = float("-inf")
            uniform = None if uniforms is None else uniforms[:, index]
            tokens = sample_logits(logits, temperature, top_p, uniform)
            if self.end_id is not None and stop_at_end and (tokens == self.end_id).any(-1).all():
                return
            yield tokens
            feed = torch.cat([tokens, tokens], dim=0) if guided else tokens
            h, state = self.backbone.step(self.embed_tokens(feed), state)


BACKBONE_FACTORIES = {
    Backbone.MAMBA: MambaBackbone,
    Backbone.TRANSFORMER: TransformerBackbone,
}


def make_backbone(cfg: GeneratorConfig) -> nn.Module:
    build = BACKBONE_FACTORIES.get(Backbone(cfg.backbone))
    if build is None:
        raise ValueError(
            f"unknown backbone {cfg.backbone!r} (expected one of {[b.value for b in Backbone]})"
        )
    return build(cfg)


def rollout_uniforms(
    seeds: Sequence[int], steps: int, codebooks: int, device: torch.device | str
) -> torch.Tensor:
    rows = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(int(seed) % (2**63 - 1))
        rows.append(torch.rand(steps, codebooks, 1, generator=generator))
    return torch.stack(rows).to(device)


def sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    uniform: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)

    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(-1)
    keep = cumulative - sorted_probs <= top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-8)

    if uniform is None:
        choice = torch.multinomial(sorted_probs.flatten(0, 1), 1).view(*logits.shape[:-1], 1)
    else:
        cdf = sorted_probs.cumsum(-1)
        choice = torch.searchsorted(cdf.contiguous(), uniform.contiguous())
        choice = choice.clamp_(max=cdf.size(-1) - 1)
    return sorted_idx.gather(-1, choice).squeeze(-1)


def token_ce_loss(
    logits: torch.Tensor, targets: torch.Tensor, token_lengths: torch.Tensor | None = None
) -> torch.Tensor:
    if token_lengths is None:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    valid = (
        torch.arange(targets.size(1), device=logits.device)[None, :] < token_lengths[:, None]
    ).float()[..., None]
    return (per_token * valid).sum() / valid.sum().clamp(min=1.0) / targets.size(-1)


@dataclass(frozen=True)
class GeneratorArchitecture:
    config: GeneratorConfig

    @property
    def backbone(self) -> Backbone:
        return self.config.backbone

    @property
    def parameter_label(self) -> str:
        return f"{self.backbone}-{self.config.d_model}d-{self.config.n_layers}l"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def build(self, device: str | torch.device) -> MotionGeneratorModule:
        return MotionGeneratorModule(self.config).to(device)

    @classmethod
    def resolve(
        cls,
        generator: GeneratorConfig,
        backbone: Backbone | str,
        codebook_size: int,
        num_codebooks: int,
        *,
        dropout: float | None = None,
        use_kernel: bool | None = None,
        max_seq_len: int | None = None,
    ) -> GeneratorArchitecture:
        backbone = Backbone(backbone)
        layer_count = generator.mamba_n_layers if backbone is Backbone.MAMBA else generator.n_layers
        overrides: dict[str, Any] = {
            "backbone": backbone,
            "n_layers": layer_count,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
        }
        if dropout is not None:
            overrides["dropout"] = dropout
        if use_kernel is not None:
            overrides["use_kernel"] = use_kernel
        if max_seq_len is not None:
            overrides["max_seq_len"] = max_seq_len
        return cls(replace(generator, **overrides))
