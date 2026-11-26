from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from text2motion.generation.contracts import LossWeights as LossWeights
from text2motion.motion.recovery import (
    recover_from_ric,
    recover_from_rot,
)
from text2motion.motion.representation import (
    FOOT_CONTACT_JOINTS,
    JOINTS,
    SLICES,
)
from text2motion.tokenization.nn_tokenizer import TokenizerNetwork


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


def codebook_tables(tokenizer: TokenizerNetwork, device: torch.device) -> list[torch.Tensor]:
    tables = []
    for unit in tokenizer.quantizer_stack().quantizers:
        all_idx = torch.arange(unit.codebook_size, device=device)
        tables.append(unit.indices_to_codes(all_idx))
    return tables


def soft_decode(logits: torch.Tensor, tokenizer: TokenizerNetwork) -> torch.Tensor:
    tables = codebook_tables(tokenizer, logits.device)

    expected_codes: list[torch.Tensor] = []
    for token_index, codebook in enumerate(tables):
        real = logits[..., token_index, : codebook.size(0)]
        expected_codes.append(real.softmax(-1) @ codebook)

    soft_codes = tokenizer.quantizer_stack().combine_code_vectors(expected_codes)
    return tokenizer.decoder(tokenizer.post_q(soft_codes))


def frame_mask(recon: torch.Tensor, frame_lengths: torch.Tensor | None) -> torch.Tensor | None:
    if frame_lengths is None:
        return None
    return (
        torch.arange(recon.size(1), device=recon.device)[None, :] < frame_lengths[:, None]
    ).float()[..., None]


def masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(a, b)
    return (torch.abs(a - b) * mask).sum() / (mask.sum().clamp(min=1.0) * a.size(-1))


def geometric_terms(
    recon: torch.Tensor, gt: torch.Tensor, frame_lengths: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    mask = frame_mask(recon, frame_lengths)
    return {name: masked_l1(recon[..., sl], gt[..., sl], mask) for name, sl in SLICES.items()}


def fk_consistency_terms(
    recon: torch.Tensor,
    gt: torch.Tensor,
    frame_lengths: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    skeleton,
) -> dict[str, torch.Tensor]:
    def recover_rot_batched(motion_raw: torch.Tensor, skeleton) -> torch.Tensor:
        return torch.stack(
            [recover_from_rot(motion_raw[b], JOINTS, skeleton) for b in range(motion_raw.size(0))]
        )

    recon_raw = recon * std + mean
    gt_raw = gt * std + mean
    pos_ric = recover_from_ric(recon_raw, JOINTS)
    pos_gt = recover_from_ric(gt_raw, JOINTS)
    skeleton.offsets.get_offsets_joints(pos_gt[0, 0].detach())
    pos_rot = recover_rot_batched(recon_raw, skeleton)

    mask = frame_mask(recon, frame_lengths)
    pmask = None if mask is None else mask.unsqueeze(-1)
    return {
        "fk_self": masked_l1(pos_rot, pos_ric, pmask),
        "fk_gt": masked_l1(pos_rot, pos_gt, pmask),
    }


def dynamics_terms(
    recon: torch.Tensor,
    gt: torch.Tensor,
    frame_lengths: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    recon_positions = recover_from_ric(recon * std + mean, JOINTS)
    gt_raw = gt * std + mean
    gt_positions = recover_from_ric(gt_raw, JOINTS)
    mask = frame_mask(recon, frame_lengths)
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


############################################################
#       UncertaintyWeighter
############################################################
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


############################################################
#       LossBatch
############################################################
@dataclass(frozen=True)
class LossBatch:
    logits: torch.Tensor
    target_tokens: torch.Tensor
    gt_motion: torch.Tensor
    lengths: torch.Tensor | None = None
    token_lengths: torch.Tensor | None = None


############################################################
#       LossSetup
############################################################
@dataclass(frozen=True)
class LossSetup:
    tokenizer: TokenizerNetwork
    weights: LossWeights
    downsample: int
    has_end: bool = False
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    skeleton: Any = None
    weighter: UncertaintyWeighter | None = None


def token_lengths_for(batch: LossBatch, setup: LossSetup) -> torch.Tensor | None:
    if batch.token_lengths is None and batch.lengths is not None:
        return (batch.lengths // setup.downsample).clamp(max=batch.target_tokens.size(1))
    return batch.token_lengths


def optional_fk_terms(
    recon: torch.Tensor, batch: LossBatch, setup: LossSetup
) -> dict[str, torch.Tensor]:
    if not (setup.weights.fk_enabled and setup.skeleton is not None):
        return {}
    if setup.mean is None or setup.std is None:
        raise ValueError("the FK losses need mean and std to recover joint positions")
    return fk_consistency_terms(
        recon,
        batch.gt_motion,
        batch.lengths,
        setup.mean,
        setup.std,
        setup.skeleton,
    )


def optional_dynamics_terms(
    recon: torch.Tensor, batch: LossBatch, setup: LossSetup
) -> dict[str, torch.Tensor]:
    if not setup.weights.dynamics_enabled:
        return {}
    if setup.mean is None or setup.std is None:
        raise ValueError("foot_skate/accel losses need mean and std to recover joint positions")
    weight_map = setup.weights.as_dict()
    dynamics = dynamics_terms(recon, batch.gt_motion, batch.lengths, setup.mean, setup.std)
    return {name: value for name, value in dynamics.items() if weight_map[name] > 0}


def weighted_total(
    ce: torch.Tensor, terms: dict[str, torch.Tensor], setup: LossSetup
) -> torch.Tensor:
    if setup.weighter is not None:
        return ce + setup.weighter(terms)
    weight_map = setup.weights.as_dict()
    return ce + sum(weight_map[name] * value for name, value in terms.items())


def generator_loss(
    batch: LossBatch, setup: LossSetup
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ce = token_ce_loss(batch.logits, batch.target_tokens, token_lengths_for(batch, setup))
    motion_logits = batch.logits[:, :-1] if setup.has_end else batch.logits
    recon = soft_decode(motion_logits, setup.tokenizer)

    terms = geometric_terms(recon, batch.gt_motion, batch.lengths)
    terms.update(optional_fk_terms(recon, batch, setup))
    terms.update(optional_dynamics_terms(recon, batch, setup))
    total = weighted_total(ce, terms, setup)

    parts = {
        "ce": ce.detach(),
        **{k: v.detach() for k, v in terms.items()},
        "total": total.detach(),
    }
    return total, parts
