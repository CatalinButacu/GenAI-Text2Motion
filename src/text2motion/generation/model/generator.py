from __future__ import annotations

from collections import deque
from collections.abc import Callable

import torch
from torch import nn

from text2motion.generation.contracts import (
    DEFAULT_SAMPLING,
    NO_ANCHORS,
    GeneratorConfig,
    SamplingConfig,
    StreamAnchors,
)
from text2motion.generation.model.backbones import instantiate_backbone
from text2motion.generation.model.decode_step import CapturedDecodeStep, EagerDecodeStep
from text2motion.generation.model.layers import init_reference_embedding, zero_linear_bias
from text2motion.generation.model.mamba import MambaBackbone
from text2motion.generation.model.sampling import sample_logits
from text2motion.generation.model.transformer import TransformerBackbone


############################################################
#       MotionTokenGenerator
############################################################
class MotionTokenGenerator(nn.Module):
    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        if cfg.per_step_text and cfg.text_prefix_len != 1:
            raise ValueError(
                f"per_step_text adds the one-vector text prefix to every token; "
                f"text_prefix_len must be 1, got {cfg.text_prefix_len}"
            )
        self.cfg = cfg
        self.end_id = cfg.codebook_size if cfg.use_end_token else None
        vocab = cfg.codebook_size + (1 if cfg.use_end_token else 0)
        token_emb = [nn.Embedding(vocab, cfg.d_model) for _ in range(cfg.num_codebooks)]
        heads = [nn.Linear(cfg.d_model, vocab) for _ in range(cfg.num_codebooks)]
        for embedding in token_emb:
            init_reference_embedding(embedding)
        for head in heads:
            zero_linear_bias(head)
        self.token_emb = nn.ModuleList(token_emb)
        self.text_proj = nn.Linear(cfg.d_text, cfg.d_model)
        zero_linear_bias(self.text_proj)
        self.heads = nn.ModuleList(heads)
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

    def decode_step_for(self, state: list, batch: int, text_vector: torch.Tensor):
        if self.captured_steps is None:
            return EagerDecodeStep(self, state, text_vector)
        if batch not in self.captured_steps:
            self.captured_steps[batch] = CapturedDecodeStep(self, batch, text_vector)
        captured = self.captured_steps[batch]
        captured.load_state(state)
        captured.load_text_vector(text_vector)
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

    def per_step_text_vector(self, text_emb: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        if not self.cfg.per_step_text:
            return torch.zeros_like(prefix)
        present = text_emb.reshape(prefix.size(0), -1).ne(0).any(-1)
        return prefix * present[:, None, None].to(prefix.dtype)

    def add_per_step_text(self, embedded: torch.Tensor, text_vector: torch.Tensor) -> torch.Tensor:
        if not self.cfg.per_step_text:
            return embedded
        if embedded.dim() == 2:
            return embedded + text_vector[:, 0]
        return embedded + text_vector

    def predict_token_logits(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(h) for head in self.heads], dim=-2)

    def forward(self, tokens: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        prefix = self.project_text_prefix(text_emb)
        text_vector = self.per_step_text_vector(text_emb, prefix)
        emb = self.add_per_step_text(self.embed_motion_tokens(tokens), text_vector)
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
        sampling: SamplingConfig = DEFAULT_SAMPLING,
        uniforms: torch.Tensor | None = None,
        anchors: StreamAnchors = NO_ANCHORS,
    ):
        device = text_emb.device
        guided = sampling.cfg_scale != 1.0
        if (anchors.text_at is not None or anchors.context_at is not None) and not anchors.every:
            raise ValueError(
                "text_at and anchor_context_at require re-anchoring; set anchor_every above 0"
            )
        text_for = anchors.text_at or (lambda index: text_emb)

        def with_null_rows(text: torch.Tensor) -> torch.Tensor:
            if guided:
                return torch.cat([text, torch.zeros_like(text)], dim=0)
            return text

        streamed_text = with_null_rows(text_emb)
        prefix = self.project_text_prefix(streamed_text)
        text_vector = self.per_step_text_vector(streamed_text, prefix)
        batch = prefix.size(0)
        self.check_stream_budget(
            steps, prefix.size(1), uniforms, anchors.every, anchors.context if anchors.every else 0
        )

        def start_from(context: list[torch.Tensor]):
            backbone = self.backbone
            if isinstance(backbone, TransformerBackbone):
                embedded = [
                    self.add_per_step_text(self.embed_motion_tokens(feed), text_vector)
                    for feed in context
                ]
                sequence = torch.cat([prefix, *[e.unsqueeze(1) for e in embedded]], dim=1)
                h, state = backbone.prefill(sequence)
                return self.predict_token_logits(h), self.decode_step_for(state, batch, text_vector)
            state = backbone.init_state(batch, device)
            h, state = backbone.step(prefix[:, 0], state)
            for position in range(1, prefix.size(1)):
                h, state = backbone.step(prefix[:, position], state)
            step = self.decode_step_for(state, batch, text_vector)
            logits = self.predict_token_logits(h)
            for feed in context:
                logits = step(feed)
            return logits, step

        recent: deque[torch.Tensor] = deque(maxlen=anchors.context if anchors.every else 0)
        logits, decode_step = start_from([])
        since_anchor = 0
        index = 0
        while steps == 0 or index < steps:
            if guided:
                cond, uncond = logits.chunk(2, dim=0)
                logits = uncond + sampling.cfg_scale * (cond - uncond)
            if self.end_id is not None and (not sampling.stop_at_end or index == 0):
                logits[..., self.end_id] = float("-inf")
            uniform = None if uniforms is None else uniforms[:, index]
            tokens = sample_logits(logits, sampling.temperature, sampling.top_p, uniform)
            if (
                self.end_id is not None
                and sampling.stop_at_end
                and (tokens == self.end_id).any(-1).all()
            ):
                return
            yield tokens
            index += 1
            if steps != 0 and index >= steps:
                return
            feed = torch.cat([tokens, tokens], dim=0) if guided else tokens
            if not anchors.every:
                logits = decode_step(feed)
                continue
            recent.append(feed)
            since_anchor += 1
            if since_anchor < anchors.every:
                logits = decode_step(feed)
                continue
            streamed_text = with_null_rows(text_for(index))
            prefix = self.project_text_prefix(streamed_text)
            text_vector = self.per_step_text_vector(streamed_text, prefix)
            context = self.reset_context(recent, index, anchors.context, anchors.context_at)
            logits, decode_step = start_from(context)
            since_anchor = 0

    def reset_context(
        self,
        recent: deque[torch.Tensor],
        index: int,
        capacity: int,
        context_at: Callable[[int], int] | None,
    ) -> list[torch.Tensor]:
        keep = capacity if context_at is None else context_at(index)
        if isinstance(keep, bool) or not isinstance(keep, int) or not 0 <= keep <= capacity:
            raise ValueError(f"reset context must be an integer within 0..{capacity}, got {keep}")
        context = list(recent)[-keep:] if keep else []
        recent.clear()
        recent.extend(context)
        return context
