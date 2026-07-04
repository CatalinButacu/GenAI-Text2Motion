"""Causal token-AR motion generator -- Contribution B.

Predicts residual-FSQ motion tokens autoregressively over (downsampled) time, conditioned on a text
embedding fed as a prefix. Two interchangeable backbones behind one interface:
  * `MambaBackbone` -- a selective SSM (S6, Gu & Dao arXiv:2312.00752). THE contribution: to our survey
    no published motion generator is a token-autoregressive S6 (all Mamba motion work is diffusion- or
    masked-bidirectional). Fixed-size recurrent state -> bounded memory while streaming.
  * `TransformerBackbone` -- a causal decoder (T2M-GPT mold, arXiv:2301.06052). The controlled twin;
    streams with a KV-cache that GROWS with sequence length.

Each backbone implements `forward(seq)` (parallel, training) and `step(x_t, state)` (recurrent,
streaming) that SHARE the same recurrence, so streamed logits == batched logits by construction
(the bounded-memory claim is then just a property of the state shape). Built from `GeneratorCfg`.
See ADR 0002 and `.claude/docs/references.md`. (Generator code is implementation behind the ADR-0002
gate; the empirical FID-vs-twin validation runs once real tokens exist.)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from text2motion.shared.config import GeneratorCfg


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# --------------------------------------------------------------------------------------------------
# Mamba (S6) backbone -- Contribution B
# --------------------------------------------------------------------------------------------------


def _parallel_scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of the first-order linear recurrence st = at*st-1 + xt (s-1 = 0) over dim 1,
    via Hillis-Steele in log2(L) affine-composition passes (Heinsen 2023, arXiv:2311.06281). Diagonal
    at -> all elementwise. Replaces the per-timestep python loop; matches step() within fp tolerance."""
    length = a.size(1)
    shift = 1
    while shift < length:
        pad = (0, 0, 0, 0, shift, 0)  # left-pad dim 1 by `shift` with identity (a=1, x=0)
        a_prev = F.pad(a[:, :-shift], pad, value=1.0)
        x_prev = F.pad(x[:, :-shift], pad, value=0.0)
        x = x + a * x_prev
        a = a * a_prev
        shift *= 2
    return x


class MambaMixer(nn.Module):
    """Minimal selective-SSM mixer with a shared step()/scan recurrence (so stream == batch)."""

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

    def step(
        self, u_t: torch.Tensor, ssm_state: torch.Tensor, conv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, z = self.in_proj(u_t).chunk(2, dim=-1)  # each (B, d_inner)
        conv_in = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)  # (B, d_inner, d_conv)
        weight = self.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
        x_conv = (conv_in * weight).sum(-1) + self.conv1d.bias
        new_conv = conv_in[..., 1:]
        x_c = F.silu(x_conv)

        a = -torch.exp(self.a_log)  # (d_inner, d_state)
        dt, b, c = torch.split(self.x_proj(x_c), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, d_inner)
        da = torch.exp(dt.unsqueeze(-1) * a)  # (B, d_inner, d_state)
        dbx = dt.unsqueeze(-1) * b.unsqueeze(1) * x_c.unsqueeze(-1)
        new_ssm = da * ssm_state + dbx
        y = (new_ssm * c.unsqueeze(1)).sum(-1) + self.d_skip * x_c
        out = self.out_proj(y * F.silu(z))
        return out, new_ssm, new_conv

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) -> (B, L, d_model). Vectorizes the projections + causal conv over the whole
        sequence (one matmul each, not one per timestep) and runs ONLY the cheap elementwise state
        recurrence sequentially. Numerically identical to looping step() -- verified by
        test_parity_mamba -- but tractable to train (the per-timestep matmul loop OOMs / is ~100x
        slower). Selective SSM of Gu & Dao (arXiv:2312.00752)."""
        batch, length, _ = seq.shape
        x, z = self.in_proj(seq).chunk(2, dim=-1)  # (B, L, d_inner) each

        x_pad = F.pad(x.transpose(1, 2), (self.d_conv - 1, 0))  # left-pad == step's zero conv_state
        x_c = F.silu(self.conv1d(x_pad).transpose(1, 2))  # (B, L, d_inner), causal

        a = -torch.exp(self.a_log)  # (d_inner, d_state)
        dt_raw, b, c = torch.split(
            self.x_proj(x_c), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        if self.use_kernel:  # fused CUDA scan (mamba-ssm); import failure is LOUD -- no fallback
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            y = selective_scan_fn(
                x_c.transpose(1, 2).contiguous(),  # u (B, d_inner, L)
                self.dt_proj(dt_raw).transpose(1, 2).contiguous(),  # delta, pre-softplus
                a,  # A (d_inner, d_state)
                b.transpose(1, 2).contiguous(),  # B (B, d_state, L)
                c.transpose(1, 2).contiguous(),  # C (B, d_state, L)
                self.d_skip,  # D (d_inner,)
                delta_softplus=True,  # kernel applies the softplus the eager path does
            ).transpose(1, 2)
            return self.out_proj(y * F.silu(z))

        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)
        da = torch.exp(dt.unsqueeze(-1) * a)  # (B, L, d_inner, d_state)  = at
        dbx = (
            dt.unsqueeze(-1) * b.unsqueeze(2) * x_c.unsqueeze(-1)
        )  # (B, L, d_inner, d_state)  = xt

        ssm = _parallel_scan(da, dbx)  # st = at*st-1 + xt, in log2(L) passes (s-1 = 0)
        y = (ssm * c.unsqueeze(2)).sum(-1) + self.d_skip * x_c  # (B, L, d_inner)
        return self.out_proj(y * F.silu(z))


class MambaBlock(nn.Module):
    def __init__(self, cfg: GeneratorCfg) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MambaMixer(
            cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand, cfg.dt_rank, cfg.use_kernel
        )
        self.drop = nn.Dropout(
            cfg.dropout
        )  # train-only -> identity in eval, so stream==batch parity holds

    def init_state(self, batch: int, device: torch.device):
        return self.mixer.init_state(batch, device)

    def step(self, x_t: torch.Tensor, state):
        out, ssm, conv = self.mixer.step(self.norm(x_t), state[0], state[1])
        return x_t + out, (ssm, conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mixer(self.norm(x)))


class MambaBackbone(nn.Module):
    def __init__(self, cfg: GeneratorCfg) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        # the fused kernel has an efficient backward of its own; checkpointing only pays for the
        # memory-hungry eager scan (the 4GB-GPU path)
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
                seq = checkpoint(block, seq, use_reentrant=False)  # recompute the eager scan in
                # backward -> bounded train memory on the 4GB GPU
            else:
                seq = block(seq)

        return self.norm_f(seq)


# --------------------------------------------------------------------------------------------------
# Transformer backbone -- the controlled twin (KV-cache grows with T)
# --------------------------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GeneratorCfg) -> None:
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
        self.drop = nn.Dropout(
            cfg.dropout
        )  # residual dropout; eval -> identity (step() path is dropout-free)
        self.attn_drop = cfg.dropout

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, dh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (self.split_heads(t) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_drop if self.training else 0.0
        )
        a = a.transpose(1, 2).reshape(x.shape)
        x = x + self.drop(self.proj(a))
        return x + self.drop(self.mlp(self.norm2(x)))

    def step(self, x_t: torch.Tensor, cache):
        q, k, v = (self.split_heads(t.unsqueeze(1)) for t in self.qkv(self.norm1(x_t)).chunk(3, -1))

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )  # query attends all cached keys
        a = a.transpose(1, 2).reshape(x_t.shape)
        x_t = x_t + self.proj(a)
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, (k, v)


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: GeneratorCfg) -> None:
        super().__init__()
        self.pos = nn.Embedding(cfg.max_seq_len + 1, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

    def init_state(self, batch: int, device: torch.device) -> list:
        return [None for _ in self.blocks]  # per-layer (k, v) caches; position is len of cache

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


# --------------------------------------------------------------------------------------------------
# Generator: token embeddings + text prefix + per-codebook heads
# --------------------------------------------------------------------------------------------------


class MotionGenerator(nn.Module):
    def __init__(self, cfg: GeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        # END is one extra id past the tokenizer vocab; the tokenizer must never decode it
        self.end_id = cfg.codebook_size if cfg.use_end_token else None
        vocab = cfg.codebook_size + (1 if cfg.use_end_token else 0)
        self.token_emb = nn.ModuleList(
            [nn.Embedding(vocab, cfg.d_model) for _ in range(cfg.num_codebooks)]
        )
        self.text_proj = nn.Linear(cfg.d_text, cfg.d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, vocab) for _ in range(cfg.num_codebooks)]
        )
        self.emb_drop = nn.Dropout(cfg.dropout)  # input/embedding dropout, shared by both backbones
        self.backbone = make_backbone(cfg)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (..., num_codebooks) -> (..., d_model), summed over codebooks."""
        out = self.token_emb[0](tokens[..., 0])

        for r in range(1, self.cfg.num_codebooks):
            out = out + self.token_emb[r](tokens[..., r])

        return out

    def text_prefix(self, text_emb: torch.Tensor) -> torch.Tensor:
        """text_emb (B, d_text) or (B, P, d_text) -> (B, P, d_model); P = cfg.text_prefix_len."""
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)
        if text_emb.size(1) != self.cfg.text_prefix_len:
            raise ValueError(
                f"text prefix has {text_emb.size(1)} tokens, cfg.text_prefix_len is "
                f"{self.cfg.text_prefix_len} (TextEncoderCfg.prefix_len must match)"
            )
        return self.text_proj(text_emb)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """h (..., d_model) -> (..., num_codebooks, vocab)."""
        return torch.stack([head(h) for head in self.heads], dim=-2)

    def forward(self, tokens: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """tokens (B, L, R), text_emb (B, d_text) or (B, P, d_text) -> logits (B, L, R, vocab).
        Teacher forced: the P prefix tokens then tokens[:, :i] predict tokens[:, i]."""
        emb = self.embed_tokens(tokens)  # (B, L, d_model)
        prefix = self.text_prefix(text_emb)  # (B, P, d_model)
        seq_in = torch.cat([prefix, emb[:, :-1]], dim=1)  # (B, P+L-1, d_model)
        h = self.backbone(self.emb_drop(seq_in))[:, prefix.size(1) - 1 :]  # (B, L, d_model)
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
    ):
        """Yield one (B, R) token step at a time, streaming with bounded state (Mamba) / KV (twin).

        cfg_scale > 1 applies classifier-free guidance: a second, unconditional row (zeroed text,
        matching the training-time `drop_text` null condition) runs in the same batch and the logits
        are extrapolated `uncond + cfg_scale * (cond - uncond)`. State stays bounded (2B rows).

        With an END-token model, `stop_at_end=False` (the GT-length eval protocol) masks END so it
        can never be sampled; `stop_at_end=True` stops as soon as every row emits END on any
        codebook (END itself is not yielded), with `steps` as the hard cap."""
        device = text_emb.device
        guided = cfg_scale != 1.0
        if guided:
            text_emb = torch.cat([text_emb, torch.zeros_like(text_emb)], dim=0)
        state = self.backbone.init_state(text_emb.size(0), device)
        prefix = self.text_prefix(text_emb)  # (2B or B, P, d_model)
        for position in range(prefix.size(1)):
            h, state = self.backbone.step(prefix[:, position], state)

        for _ in range(steps):
            logits = self.logits(h)
            if guided:
                cond, uncond = logits.chunk(2, dim=0)
                logits = uncond + cfg_scale * (cond - uncond)
            if self.end_id is not None and not stop_at_end:
                logits[..., self.end_id] = float("-inf")  # fixed-length protocol: END unreachable
            tokens = sample_logits(logits, temperature, top_p)  # (B, R)
            if self.end_id is not None and stop_at_end and (tokens == self.end_id).any(-1).all():
                return
            yield tokens
            feed = torch.cat([tokens, tokens], dim=0) if guided else tokens
            h, state = self.backbone.step(self.embed_tokens(feed), state)


def make_backbone(cfg: GeneratorCfg) -> nn.Module:
    if cfg.backbone == "mamba":
        return MambaBackbone(cfg)

    if cfg.backbone == "transformer":
        return TransformerBackbone(cfg)

    raise ValueError(f"unknown backbone {cfg.backbone!r} (expected 'mamba' or 'transformer')")


def sample_logits(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Nucleus sample per codebook. logits (B, R, V) -> tokens (B, R). Non-greedy by default
    (greedy collapses -- the T2M-GPT failure mode)."""
    if temperature <= 0:
        return logits.argmax(-1)

    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(-1)
    keep = cumulative - sorted_probs <= top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-8)
    choice = torch.multinomial(sorted_probs.flatten(0, 1), 1).view(*logits.shape[:-1], 1)
    return sorted_idx.gather(-1, choice).squeeze(-1)


def token_ce_loss(
    logits: torch.Tensor, targets: torch.Tensor, token_lengths: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean cross-entropy over codebooks. logits (B, L, R, V), targets (B, L, R). When token_lengths
    (B,) is given, padded time positions (>= length) are masked out of the mean."""
    if token_lengths is None:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(targets.shape)  # (B, L, R)
    valid = (
        torch.arange(targets.size(1), device=logits.device)[None, :] < token_lengths[:, None]
    ).float()[..., None]  # (B, L, 1)
    return (per_token * valid).sum() / valid.sum().clamp(min=1.0) / targets.size(-1)
