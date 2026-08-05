import torch
from torch import nn
from torch.nn import functional as F

from text2motion.data.hml3d.feature import recover_from_ric, recover_from_rot
from text2motion.model.generator import token_ce_loss
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import TrainCfg

SLICE_ROOT = slice(0, 4)
SLICE_RIC = slice(4, 67)
SLICE_ROT6D = slice(67, 193)
SLICE_VEL = slice(193, 259)
SLICE_FOOT = slice(259, 263)
JOINTS_NUM = 22

GEO_TERMS = ("root", "ric", "rot6d", "vel", "foot")
FK_TERMS = ("fk_self", "fk_gt")


def codebook_tables(tokenizer: ResidualFsqTokenizer, device: torch.device) -> list[torch.Tensor]:
    cached = getattr(tokenizer, "_codebook_tables", None)
    if cached is not None and cached[0].device == device:
        return cached

    tables = []
    for unit in tokenizer.quantizer.units:
        all_idx = torch.arange(unit.codebook_size, device=device)
        tables.append(unit.indices_to_codes(all_idx))
    tokenizer._codebook_tables = tables
    return tables


def soft_decode(logits: torch.Tensor, tokenizer: ResidualFsqTokenizer) -> torch.Tensor:
    tables = codebook_tables(tokenizer, logits.device)

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


def geometric_terms(
    recon: torch.Tensor, gt: torch.Tensor, frame_lengths: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    mask = _frame_mask(recon, frame_lengths)
    return {
        "root": _masked_l1(recon[..., SLICE_ROOT], gt[..., SLICE_ROOT], mask),
        "ric": _masked_l1(recon[..., SLICE_RIC], gt[..., SLICE_RIC], mask),
        "rot6d": _masked_l1(recon[..., SLICE_ROT6D], gt[..., SLICE_ROT6D], mask),
        "vel": _masked_l1(recon[..., SLICE_VEL], gt[..., SLICE_VEL], mask),
        "foot": _masked_l1(recon[..., SLICE_FOOT], gt[..., SLICE_FOOT], mask),
    }


def _recover_rot_batched(motion_raw: torch.Tensor, skeleton) -> torch.Tensor:
    return torch.stack(
        [recover_from_rot(motion_raw[b], JOINTS_NUM, skeleton) for b in range(motion_raw.size(0))]
    )


def fk_consistency_terms(
    recon: torch.Tensor,
    gt: torch.Tensor,
    frame_lengths: torch.Tensor | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    skeleton,
) -> dict[str, torch.Tensor]:
    recon_raw = recon * std + mean
    gt_raw = gt * std + mean
    pos_ric = recover_from_ric(recon_raw, JOINTS_NUM)  # (B, T, J, 3)
    pos_gt = recover_from_ric(gt_raw, JOINTS_NUM)
    skeleton.get_offsets_joints(pos_gt[0, 0].detach())
    pos_rot = _recover_rot_batched(recon_raw, skeleton)  # (B, T, J, 3)

    mask = _frame_mask(recon, frame_lengths)
    pmask = None if mask is None else mask.unsqueeze(-1)  # (B, T, 1, 1) over (J, 3)
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


def active_term_names(cfg: TrainCfg) -> tuple[str, ...]:
    weights = term_weights(cfg)
    return tuple(name for name in (*GEO_TERMS, *FK_TERMS) if weights[name] > 0)


def term_weights(cfg: TrainCfg) -> dict[str, float]:
    return {
        "root": cfg.w_root,
        "ric": cfg.w_ric,
        "rot6d": cfg.w_rot6d,
        "vel": cfg.w_vel,
        "foot": cfg.w_foot,
        "fk_self": cfg.w_fk_self,
        "fk_gt": cfg.w_fk_gt,
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
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    skeleton=None,
    weighter: UncertaintyWeighter | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if token_lengths is None and lengths is not None:
        token_lengths = (lengths // tokenizer.cfg.downsample).clamp(max=target_tokens.size(1))

    ce = token_ce_loss(logits, target_tokens, token_lengths)
    motion_logits = logits[:, :-1] if has_end else logits  # END slot decodes no motion
    recon = soft_decode(motion_logits, tokenizer)

    terms = geometric_terms(recon, gt_motion, lengths)
    weights = term_weights(cfg)
    if (weights["fk_self"] > 0 or weights["fk_gt"] > 0) and skeleton is not None:
        terms.update(fk_consistency_terms(recon, gt_motion, lengths, mean, std, skeleton))

    if weighter is not None:
        total = ce + weighter(terms)
    else:
        total = ce + sum(weights[name] * value for name, value in terms.items())

    parts = {"ce": ce.item(), **{k: v.item() for k, v in terms.items()}, "total": total.item()}
    return total, parts
