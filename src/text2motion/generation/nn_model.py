from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.generation.contracts import (
    Backbone,
    GeneratorConfig,
)
from text2motion.generation.contracts import BackboneChoice as BackboneChoice
from text2motion.generation.nn_backbones import (
    MambaBackbone,
    TransformerBackbone,
    instantiate_backbone,
)


class MotionTokenGenerator(nn.Module):
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
        self.backbone = instantiate_backbone(cfg)
        self.captured_steps: dict[int, CapturedDecodeStep] | None = None

    def enable_cuda_graph_decode(self) -> None:
        if not isinstance(self.backbone, MambaBackbone):
            raise ValueError(
                f"cuda-graph decode needs the fixed-size recurrent state of the mamba backbone, "
                f"got {type(self.backbone).__name__}; set service.cuda_graph_decode: false"
            )
        device = next(self.parameters()).device
        if device.type != "cuda":
            raise ValueError(
                f"cuda-graph decode needs a cuda device, the generator is on {device}; "
                "set service.cuda_graph_decode: false"
            )
        self.captured_steps = {}

    def decode_step_for(self, state: list, batch: int):
        if self.captured_steps is None:
            return EagerDecodeStep(self, state)
        if batch not in self.captured_steps:
            self.captured_steps[batch] = CapturedDecodeStep(self, batch)
        captured = self.captured_steps[batch]
        captured.load_state(state)
        return captured

    def embed_motion_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.token_emb[0](tokens[..., 0])

        for r in range(1, self.cfg.num_codebooks):
            out = out + self.token_emb[r](tokens[..., r])

        return out

    def project_text_prefix(self, text_emb: torch.Tensor) -> torch.Tensor:
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)
        if text_emb.size(1) != self.cfg.text_prefix_len:
            raise ValueError(
                f"text prefix has {text_emb.size(1)} tokens, cfg.text_prefix_len is "
                f"{self.cfg.text_prefix_len} (TextEncoderCfg.prefix_len must match)"
            )
        return self.text_proj(text_emb)

    def predict_token_logits(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(h) for head in self.heads], dim=-2)

    def forward(self, tokens: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        emb = self.embed_motion_tokens(tokens)
        prefix = self.project_text_prefix(text_emb)
        seq_in = torch.cat([prefix, emb[:, :-1]], dim=1)
        h = self.backbone(self.emb_drop(seq_in))[:, prefix.size(1) - 1 :]
        return self.predict_token_logits(h)

    def check_stream_budget(
        self, steps: int, prefix_len: int, uniforms, anchor_every: int = 0, anchor_context: int = 0
    ) -> None:
        if steps < 0:
            raise ValueError(f"steps must be >= 0 (0 means unbounded), got {steps}")
        if steps == 0 and uniforms is not None:
            raise ValueError("unbounded streaming cannot consume a fixed uniforms tensor")
        if anchor_every < 0 or anchor_context < 0:
            raise ValueError(
                f"anchor_every ({anchor_every}) and anchor_context ({anchor_context}) must be >= 0"
            )
        limit = getattr(self.backbone, "max_positions", None)
        if limit is None:
            return
        available = limit - prefix_len
        if anchor_every:
            deepest = anchor_context + anchor_every
            if deepest > available:
                raise ValueError(
                    f"re-anchoring every {anchor_every} tokens over a {anchor_context}-token "
                    f"context reaches position {prefix_len + deepest}, but "
                    f"{type(self.backbone).__name__} has {limit} learned positions"
                )
            return
        if steps == 0:
            raise RuntimeError(
                f"{type(self.backbone).__name__} cannot stream unboundedly: it has "
                f"{limit} learned positions, so generation stops after {available} steps. "
                "Unbounded streaming requires a recurrent backbone (mamba); "
                "for this backbone pass a positive step budget instead."
            )
        if steps > available:
            raise ValueError(
                f"{type(self.backbone).__name__} supports at most {available} steps after a "
                f"{prefix_len}-token text prefix ({limit} learned positions), got {steps}"
            )

    @torch.no_grad()
    def stream_token_indices(
        self,
        text_emb: torch.Tensor,
        steps: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        cfg_scale: float = 1.0,
        stop_at_end: bool = False,
        uniforms: torch.Tensor | None = None,
        anchor_every: int = 0,
        anchor_context: int = 16,
        text_at: Callable[[int], torch.Tensor] | None = None,
    ):
        device = text_emb.device
        guided = cfg_scale != 1.0
        if text_at is not None and not anchor_every:
            raise ValueError("text_at changes the text at a reset; set anchor_every above 0")
        text_for = text_at or (lambda index: text_emb)

        def prefix_of(text: torch.Tensor) -> torch.Tensor:
            if guided:
                text = torch.cat([text, torch.zeros_like(text)], dim=0)
            return self.project_text_prefix(text)

        prefix = prefix_of(text_emb)
        batch = prefix.size(0)
        self.check_stream_budget(
            steps, prefix.size(1), uniforms, anchor_every, anchor_context if anchor_every else 0
        )

        def start_from(context: list[torch.Tensor]):
            backbone = self.backbone
            if isinstance(backbone, TransformerBackbone):
                embedded = [self.embed_motion_tokens(feed) for feed in context]
                sequence = torch.cat([prefix, *[e.unsqueeze(1) for e in embedded]], dim=1)
                h, state = backbone.prefill(sequence)
                return self.predict_token_logits(h), self.decode_step_for(state, batch)
            state = backbone.init_state(batch, device)
            h, state = backbone.step(prefix[:, 0], state)
            for position in range(1, prefix.size(1)):
                h, state = backbone.step(prefix[:, position], state)
            step = self.decode_step_for(state, batch)
            logits = self.predict_token_logits(h)
            for feed in context:
                logits = step(feed)
            return logits, step

        recent: deque[torch.Tensor] = deque(maxlen=anchor_context if anchor_every else 0)
        logits, decode_step = start_from([])
        since_anchor = 0
        index = 0
        while steps == 0 or index < steps:
            if guided:
                cond, uncond = logits.chunk(2, dim=0)
                logits = uncond + cfg_scale * (cond - uncond)
            if self.end_id is not None and (not stop_at_end or index == 0):
                logits[..., self.end_id] = float("-inf")
            uniform = None if uniforms is None else uniforms[:, index]
            tokens = sample_logits(logits, temperature, top_p, uniform)
            if self.end_id is not None and stop_at_end and (tokens == self.end_id).any(-1).all():
                return
            yield tokens
            index += 1
            if steps != 0 and index >= steps:
                return
            feed = torch.cat([tokens, tokens], dim=0) if guided else tokens
            if not anchor_every:
                logits = decode_step(feed)
                continue
            recent.append(feed)
            since_anchor += 1
            if since_anchor < anchor_every:
                logits = decode_step(feed)
                continue
            prefix = prefix_of(text_for(index))
            logits, decode_step = start_from(list(recent))
            since_anchor = 0


class EagerDecodeStep:
    def __init__(self, model: MotionTokenGenerator, state: list) -> None:
        self.model = model
        self.state = state

    def __call__(self, feed: torch.Tensor) -> torch.Tensor:
        embedded = self.model.embed_motion_tokens(feed)
        h, self.state = self.model.backbone.step(embedded, self.state)
        return self.model.predict_token_logits(h)


class CapturedDecodeStep:
    warmup_steps = 3

    def __init__(self, model: MotionTokenGenerator, batch: int) -> None:
        device = next(model.parameters()).device
        codebooks = model.cfg.num_codebooks
        self.tokens = torch.zeros(batch, codebooks, dtype=torch.long, device=device)
        self.state = model.backbone.init_state(batch, device)
        side_stream = torch.cuda.Stream(device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(self.warmup_steps):
                EagerDecodeStep(model, list(self.state))(self.tokens)
        torch.cuda.current_stream(device).wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            embedded = model.embed_motion_tokens(self.tokens)
            h, next_state = model.backbone.step(embedded, list(self.state))
            for (ssm, conv), (next_ssm, next_conv) in zip(self.state, next_state, strict=True):
                ssm.copy_(next_ssm)
                conv.copy_(next_conv)
            self.logits = model.predict_token_logits(h)

    def load_state(self, state: list) -> None:
        for (ssm, conv), (source_ssm, source_conv) in zip(self.state, state, strict=True):
            ssm.copy_(source_ssm)
            conv.copy_(source_conv)

    def __call__(self, feed: torch.Tensor) -> torch.Tensor:
        self.tokens.copy_(feed)
        self.graph.replay()
        return self.logits


def seeded_sampling_uniforms(
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
class GeneratorModelSpec:
    config: GeneratorConfig

    @property
    def backbone(self) -> Backbone:
        return self.config.backbone

    @property
    def architecture_label(self) -> str:
        return f"{self.backbone}-{self.config.d_model}d-{self.config.n_layers}l"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def create_model(self, device: str | torch.device) -> MotionTokenGenerator:
        return MotionTokenGenerator(self.config).to(device)

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
    ) -> GeneratorModelSpec:
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
