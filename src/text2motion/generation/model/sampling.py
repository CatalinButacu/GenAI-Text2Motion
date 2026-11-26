from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


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
