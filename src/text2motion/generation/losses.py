from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.generation.model import token_ce_loss
from text2motion.motion.representation import (
    FK_TERMS,
    GEO_TERMS,
    JOINTS,
    SLICES,
    recover_from_ric,
    recover_from_rot,
)
from text2motion.tokenization.model import TokenizerModule


class LossWeighting(StrEnum):
    FIXED = "fixed"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True)
class LossWeights:
    root: float = 0.3
    ric: float = 0.5
    rot6d: float = 0.5
    vel: float = 0.3
    foot: float = 0.1
    fk_self: float = 0.0
    fk_gt: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def active_terms(self) -> tuple[str, ...]:
        weights = self.as_dict()
        return tuple(name for name in (*GEO_TERMS, *FK_TERMS) if weights[name] > 0)

    @property
    def fk_enabled(self) -> bool:
        return self.fk_self > 0 or self.fk_gt > 0


def _codebook_tables(tokenizer: TokenizerModule, device: torch.device) -> list[torch.Tensor]:
    cached = getattr(tokenizer, "_codebook_tables_cache", None)
    if cached is not None and cached[0].device == device:
        return cached

    tables = []
    for unit in tokenizer.quantizer.units:
        all_idx = torch.arange(unit.codebook_size, device=device)
        tables.append(unit.indices_to_codes(all_idx))
    tokenizer._codebook_tables_cache = tables
    return tables


def soft_decode(logits: torch.Tensor, tokenizer: TokenizerModule) -> torch.Tensor:
    tables = _codebook_tables(tokenizer, logits.device)

    expected_codes: list[torch.Tensor] = []
    for token_index, codebook in enumerate(tables):
        real = logits[..., token_index, : codebook.size(0)]
        expected_codes.append(real.softmax(-1) @ codebook)

    soft_codes = tokenizer.quantizer.combine_codes(expected_codes)
    return tokenizer.decoder(tokenizer.post_q(soft_codes))


def _frame_mask(recon: torch.Tensor, frame_lengths: torch.Tensor | None) -> torch.Tensor | None:
    if frame_lengths is None:
        return None
    return (
        torch.arange(recon.size(1), device=recon.device)[None, :] < frame_lengths[:, None]
    ).float()[..., None]


def _masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(a, b)
    return (torch.abs(a - b) * mask).sum() / (mask.sum().clamp(min=1.0) * a.size(-1))


def _geometric_terms(
    recon: torch.Tensor, gt: torch.Tensor, frame_lengths: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    mask = _frame_mask(recon, frame_lengths)
    return {name: _masked_l1(recon[..., sl], gt[..., sl], mask) for name, sl in SLICES.items()}


def _recover_rot_batched(motion_raw: torch.Tensor, skeleton) -> torch.Tensor:
    return torch.stack(
        [recover_from_rot(motion_raw[b], JOINTS, skeleton) for b in range(motion_raw.size(0))]
    )


def _fk_consistency_terms(
    recon: torch.Tensor,
    gt: torch.Tensor,
    frame_lengths: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    skeleton,
) -> dict[str, torch.Tensor]:
    recon_raw = recon * std + mean
    gt_raw = gt * std + mean
    pos_ric = recover_from_ric(recon_raw, JOINTS)
    pos_gt = recover_from_ric(gt_raw, JOINTS)
    skeleton.get_offsets_joints(pos_gt[0, 0].detach())
    pos_rot = _recover_rot_batched(recon_raw, skeleton)

    mask = _frame_mask(recon, frame_lengths)
    pmask = None if mask is None else mask.unsqueeze(-1)
    return {
        "fk_self": _masked_l1(pos_rot, pos_ric, pmask),
        "fk_gt": _masked_l1(pos_rot, pos_gt, pmask),
    }


class UncertaintyWeighter(nn.Module):
    def __init__(self, term_names: tuple[str, ...]) -> None:
        super().__init__()
        self.term_names = term_names
        self.log_vars = nn.Parameter(torch.zeros(len(term_names)))

    def forward(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        total = terms[self.term_names[0]].new_zeros(())
        for i, name in enumerate(self.term_names):
            if name in terms:
                total = total + torch.exp(-self.log_vars[i]) * terms[name] + self.log_vars[i]
        return total


def generator_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    gt_motion: torch.Tensor,
    tokenizer: TokenizerModule,
    weights: LossWeights,
    downsample: int,
    lengths: torch.Tensor | None = None,
    token_lengths: torch.Tensor | None = None,
    has_end: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    skeleton=None,
    weighter: UncertaintyWeighter | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if token_lengths is None and lengths is not None:
        token_lengths = (lengths // downsample).clamp(max=target_tokens.size(1))

    ce = token_ce_loss(logits, target_tokens, token_lengths)
    motion_logits = logits[:, :-1] if has_end else logits
    recon = soft_decode(motion_logits, tokenizer)

    terms = _geometric_terms(recon, gt_motion, lengths)
    weight_map = weights.as_dict()
    if weights.fk_enabled and skeleton is not None:
        terms.update(_fk_consistency_terms(recon, gt_motion, lengths, mean, std, skeleton))

    if weighter is not None:
        total = ce + weighter(terms)
    else:
        total = ce + sum(weight_map[name] * value for name, value in terms.items())

    parts = {
        "ce": ce.detach(),
        **{k: v.detach() for k, v in terms.items()},
        "total": total.detach(),
    }
    return total, parts
