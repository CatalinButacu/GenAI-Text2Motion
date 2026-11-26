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
from text2motion.generation.model.layers import init_reference_embedding, zero_linear_bias
from text2motion.generation.model.streaming.budget import (
    check_anchor_reach,
    check_budget_values,
    check_step_reach,
)
from text2motion.generation.model.streaming.cuda_graph import CudaGraphDecode
from text2motion.generation.model.streaming.loop import stream_indices


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
        self.cuda_graph = CudaGraphDecode(self)

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
        check_budget_values(steps, uniforms, anchor_every, anchor_context)
        limit = getattr(self.backbone, "max_positions", None)
        if limit is None:
            return
        name = type(self.backbone).__name__
        if anchor_every:
            check_anchor_reach(name, limit, prefix_len, anchor_every, anchor_context)
            return
        check_step_reach(name, limit, prefix_len, steps)

    @torch.no_grad()
    def stream_token_indices(
        self,
        text_emb: torch.Tensor,
        steps: int,
        sampling: SamplingConfig = DEFAULT_SAMPLING,
        uniforms: torch.Tensor | None = None,
        anchors: StreamAnchors = NO_ANCHORS,
    ):
        return stream_indices(self, text_emb, steps, sampling, uniforms, anchors)

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
