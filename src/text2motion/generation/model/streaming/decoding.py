from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from text2motion.generation.contracts import SamplingConfig
from text2motion.generation.model.sampling import sample_logits
from text2motion.generation.model.transformer import TransformerBackbone

if TYPE_CHECKING:
    from text2motion.generation.model.generator import MotionTokenGenerator


def with_null_rows(text: torch.Tensor, guided: bool) -> torch.Tensor:
    if guided:
        return torch.cat([text, torch.zeros_like(text)], dim=0)
    return text


def project_text(
    model: MotionTokenGenerator, text: torch.Tensor, guided: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    streamed = with_null_rows(text, guided)
    prefix = model.project_text_prefix(streamed)
    return prefix, model.per_step_text_vector(streamed, prefix)


def start_decoding(
    model: MotionTokenGenerator,
    context: list[torch.Tensor],
    prefix: torch.Tensor,
    text_vector: torch.Tensor,
    batch: int,
    device: torch.device,
):
    backbone = model.backbone
    if isinstance(backbone, TransformerBackbone):
        embedded = [
            model.add_per_step_text(model.embed_motion_tokens(feed), text_vector)
            for feed in context
        ]
        sequence = torch.cat([prefix, *[e.unsqueeze(1) for e in embedded]], dim=1)
        h, state = backbone.prefill(sequence)
        return model.predict_token_logits(h), model.cuda_graph.decode_step_for(
            state, batch, text_vector
        )
    state = backbone.init_state(batch, device)
    h, state = backbone.step(prefix[:, 0], state)
    for position in range(1, prefix.size(1)):
        h, state = backbone.step(prefix[:, position], state)
    step = model.cuda_graph.decode_step_for(state, batch, text_vector)
    logits = model.predict_token_logits(h)
    for feed in context:
        logits = step(feed)
    return logits, step


def combine_guidance(logits: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    cond, uncond = logits.chunk(2, dim=0)
    return uncond + cfg_scale * (cond - uncond)


def sample_step(
    model: MotionTokenGenerator,
    logits: torch.Tensor,
    sampling: SamplingConfig,
    uniforms: torch.Tensor | None,
    index: int,
) -> torch.Tensor:
    if sampling.cfg_scale != 1.0:
        logits = combine_guidance(logits, sampling.cfg_scale)
    if model.end_id is not None and (not sampling.stop_at_end or index == 0):
        logits[..., model.end_id] = float("-inf")
    uniform = None if uniforms is None else uniforms[:, index]
    return sample_logits(logits, sampling.temperature, sampling.top_p, uniform)


def reached_end(end_id: int | None, sampling: SamplingConfig, tokens: torch.Tensor) -> bool:
    return end_id is not None and sampling.stop_at_end and bool((tokens == end_id).any(-1).all())
