"""Generator training losses — the recipe that escapes the prior plateau (token-CE only).

The key term is **soft-decode reconstruction**: softmax over each codebook's logits gives an expected
FSQ code; summed across the residual levels and run through the tokenizer's FROZEN decoder, this
yields a differentiable motion reconstruction whose gradient reaches the generator logits WITHOUT a
non-differentiable argmax. The tokenizer's params are frozen (requires_grad=False) so the decoder is
differentiable-forward but never updated. See `.claude/skills/t2m-losses` and ADR 0001/0002.
"""

import torch
from torch.nn import functional as F

from text2motion.model.generator import token_ce_loss
from text2motion.model.tokenizer import GroupedFSQ, ResidualFsqTokenizer
from text2motion.shared.config import TrainCfg

# HumanML3D-263 channel layout (Guo et al.): index 3 = root height; last 4 = binary foot contacts.
ROOT_HEIGHT_IDX = 3
FOOT_CONTACT = slice(-4, None)


def soft_decode(logits: torch.Tensor, tokenizer: ResidualFsqTokenizer) -> torch.Tensor:
    """logits (B, T', R, V) -> reconstructed motion (B, T, in_dim) via expected FSQ codes through the
    frozen decoder. Differentiable in `logits`; the tokenizer is not updated. Grouped quantizers
    CONCATENATE the per-token expected codes; residual quantizers SUM them."""
    quantizer = tokenizer.quantizer
    units = quantizer.groups if isinstance(quantizer, GroupedFSQ) else quantizer.layers

    expected_codes: list[torch.Tensor] = []
    for token_index, unit in enumerate(units):
        all_idx = torch.arange(unit.codebook_size, device=logits.device)
        codebook = unit.indices_to_codes(all_idx)  # (V, fsq_dim)
        # slice to the real vocab: an END-token generator carries one extra logit column that has
        # no code; the expectation renormalizes over decodable codes only
        real = logits[..., token_index, : unit.codebook_size]
        expected_codes.append(real.softmax(-1) @ codebook)

    if isinstance(quantizer, GroupedFSQ):
        soft_codes = torch.cat(expected_codes, dim=-1)
    else:
        soft_codes = torch.stack(expected_codes, dim=0).sum(0)

    return tokenizer.decoder(tokenizer.post_q(soft_codes))


def _masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(a, b)
    return (torch.abs(a - b) * mask).sum() / (mask.sum().clamp(min=1.0) * a.size(-1))


def geometric_losses(
    recon: torch.Tensor, gt: torch.Tensor, frame_lengths: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Motion-space terms on the soft-decoded reconstruction vs ground truth (both (B, T, 263)).
    When frame_lengths (B,) is given, padded frames (>= length) are masked out."""
    mask = None
    if frame_lengths is not None:
        mask = (
            torch.arange(recon.size(1), device=recon.device)[None, :] < frame_lengths[:, None]
        ).float()[..., None]
    vel_mask = mask[:, 1:] if mask is not None else None
    root = slice(ROOT_HEIGHT_IDX, ROOT_HEIGHT_IDX + 1)
    return {
        "recon": _masked_l1(recon, gt, mask),
        "velocity": _masked_l1(recon[:, 1:] - recon[:, :-1], gt[:, 1:] - gt[:, :-1], vel_mask),
        "foot": _masked_l1(recon[..., FOOT_CONTACT], gt[..., FOOT_CONTACT], mask),
        "root": _masked_l1(recon[..., root], gt[..., root], mask),
    }


def generator_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    gt_motion: torch.Tensor,
    tokenizer: ResidualFsqTokenizer,
    cfg: TrainCfg,
    lengths: torch.Tensor | None = None,
    token_lengths: torch.Tensor | None = None,
    has_end: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss + per-term scalars. logits (B,T',R,V), target_tokens (B,T',R), gt_motion (B,T,263).
    `lengths` (B,) gives the unpadded motion length per clip; padded tokens/frames are masked out.
    `token_lengths` overrides the derived token mask (END-token training: includes the END slot);
    `has_end` drops the appended END time position from the motion-space (soft-decode) terms."""
    if token_lengths is None and lengths is not None:
        token_lengths = (lengths // tokenizer.cfg.downsample).clamp(max=target_tokens.size(1))

    ce = token_ce_loss(logits, target_tokens, token_lengths)
    motion_logits = logits[:, :-1] if has_end else logits  # END slot decodes no motion
    geo = geometric_losses(soft_decode(motion_logits, tokenizer), gt_motion, lengths)
    total = (
        ce
        + cfg.w_recon * geo["recon"]
        + cfg.w_velocity * geo["velocity"]
        + cfg.w_foot * geo["foot"]
        + cfg.w_root * geo["root"]
    )
    parts = {"ce": ce.item(), **{k: v.item() for k, v in geo.items()}, "total": total.item()}
    return total, parts
