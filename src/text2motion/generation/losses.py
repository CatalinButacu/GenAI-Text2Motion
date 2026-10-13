from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.generation.contracts import LossWeights as LossWeights
from text2motion.generation.nn_model import token_ce_loss
from text2motion.motion.representation import (
    FOOT_CONTACT_JOINTS,
    JOINTS,
    SLICES,
    recover_from_ric,
    recover_from_rot,
)
from text2motion.tokenization.nn_model import MotionTokenizerNetwork


def __codebook_tables(
    tokenizer: MotionTokenizerNetwork, device: torch.device
) -> list[torch.Tensor]:
    cached = getattr(tokenizer, "_codebook_tables_cache", None)
    if cached is not None and cached[0].device == device:
        return cached

    tables = []
    for unit in tokenizer.quantizer.quantizer_units:
        all_idx = torch.arange(unit.codebook_size, device=device)
        tables.append(unit.indices_to_codes(all_idx))
    tokenizer._codebook_tables_cache = tables
    return tables


def soft_decode(logits: torch.Tensor, tokenizer: MotionTokenizerNetwork) -> torch.Tensor:
    tables = __codebook_tables(tokenizer, logits.device)

    expected_codes: list[torch.Tensor] = []
    for token_index, codebook in enumerate(tables):
        real = logits[..., token_index, : codebook.size(0)]
        expected_codes.append(real.softmax(-1) @ codebook)

    soft_codes = tokenizer.quantizer.compose_latent_codes(expected_codes)
    return tokenizer.decoder(tokenizer.post_q(soft_codes))


def __frame_mask(recon: torch.Tensor, frame_lengths: torch.Tensor | None) -> torch.Tensor | None:
    if frame_lengths is None:
        return None
    return (
        torch.arange(recon.size(1), device=recon.device)[None, :] < frame_lengths[:, None]
    ).float()[..., None]


def __masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(a, b)
    return (torch.abs(a - b) * mask).sum() / (mask.sum().clamp(min=1.0) * a.size(-1))


def __geometric_terms(
    recon: torch.Tensor, gt: torch.Tensor, frame_lengths: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    mask = __frame_mask(recon, frame_lengths)
    return {name: __masked_l1(recon[..., sl], gt[..., sl], mask) for name, sl in SLICES.items()}


def __recover_rot_batched(motion_raw: torch.Tensor, skeleton) -> torch.Tensor:
    return torch.stack(
        [recover_from_rot(motion_raw[b], JOINTS, skeleton) for b in range(motion_raw.size(0))]
    )


def __fk_consistency_terms(
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
    pos_rot = __recover_rot_batched(recon_raw, skeleton)

    mask = __frame_mask(recon, frame_lengths)
    pmask = None if mask is None else mask.unsqueeze(-1)
    return {
        "fk_self": __masked_l1(pos_rot, pos_ric, pmask),
        "fk_gt": __masked_l1(pos_rot, pos_gt, pmask),
    }


def __dynamics_terms(
    recon: torch.Tensor,
    gt: torch.Tensor,
    frame_lengths: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    recon_positions = recover_from_ric(recon * std + mean, JOINTS)
    gt_raw = gt * std + mean
    gt_positions = recover_from_ric(gt_raw, JOINTS)
    mask = __frame_mask(recon, frame_lengths)
    valid = torch.ones_like(recon[..., :1]) if mask is None else mask

    feet = list(FOOT_CONTACT_JOINTS)
    foot_velocity = (
        recon_positions[:, 1:, feet][..., [0, 2]] - recon_positions[:, :-1, feet][..., [0, 2]]
    )
    contact = gt_raw[:, :-1, SLICES["foot"]].clamp(0.0, 1.0) * valid[:, 1:]
    foot_skate = (foot_velocity.norm(dim=-1) * contact).sum() / contact.sum().clamp(min=1.0)

    recon_acceleration = (
        recon_positions[:, 2:] - 2 * recon_positions[:, 1:-1] + recon_positions[:, :-2]
    )
    gt_acceleration = gt_positions[:, 2:] - 2 * gt_positions[:, 1:-1] + gt_positions[:, :-2]
    acceleration_valid = (valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]).unsqueeze(-1)
    acceleration_gap = (recon_acceleration - gt_acceleration).abs() * acceleration_valid
    accel = acceleration_gap.sum() / (acceleration_valid.sum().clamp(min=1.0) * JOINTS * 3)
    return {"foot_skate": foot_skate, "accel": accel}


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
    tokenizer: MotionTokenizerNetwork,
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

    terms = __geometric_terms(recon, gt_motion, lengths)
    weight_map = weights.as_dict()
    if weights.fk_enabled and skeleton is not None:
        terms.update(__fk_consistency_terms(recon, gt_motion, lengths, mean, std, skeleton))
    if weights.dynamics_enabled:
        if mean is None or std is None:
            raise ValueError("foot_skate/accel losses need mean and std to recover joint positions")
        dynamics = __dynamics_terms(recon, gt_motion, lengths, mean, std)
        terms.update({name: value for name, value in dynamics.items() if weight_map[name] > 0})

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
