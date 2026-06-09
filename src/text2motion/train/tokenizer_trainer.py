"""Tokenizer trainer (Contribution A) — trains the Residual-FSQ tokenizer or the strong-RVQ baseline.

One step: encode→quantize→decode a batch of normalized motion windows and minimise the reconstruction
loss (feature-L1 + velocity), plus the commitment term for the RVQ baseline (FSQ needs none). EMA of
the weights is kept for eval. Codebook perplexity (effective codes used) is logged per step. The two
tokenizers share this trainer via their ``forward`` returns: FSQ -> (recon, indices), RVQ ->
(recon, indices, commit). See ``.claude/skills/motion-tokenizer`` and ``.claude/skills/t2m-losses``.
"""

import torch
from torch import nn

from text2motion.model.tokenizer import reconstruction_loss
from text2motion.train.ema import Ema


class TokenizerTrainer:
    def __init__(
        self,
        tokenizer: nn.Module,
        lr: float,
        weight_decay: float,
        ema_decay: float,
        commit_beta: float = 0.0,
        grad_clip: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.commit_beta = commit_beta
        self.grad_clip = grad_clip
        self.codebook_size = int(tokenizer.codebook_size)
        self.opt = torch.optim.AdamW(tokenizer.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema = Ema(tokenizer, ema_decay)

    def _perplexity(self, indices: torch.Tensor) -> float:
        """Mean over codebooks of exp(entropy of code usage) — effective codes used this batch."""
        perplexities = []
        for codebook in range(indices.shape[-1]):
            counts = torch.bincount(
                indices[..., codebook].reshape(-1), minlength=self.codebook_size
            ).float()
            probs = counts[counts > 0] / counts.sum()
            perplexities.append(float(torch.exp(-(probs * probs.log()).sum())))
        return sum(perplexities) / len(perplexities)

    def train_step(self, motion: torch.Tensor) -> dict[str, float]:
        """motion (B, window, 263) normalized. Returns per-term scalars."""
        out = self.tokenizer(motion)
        if len(out) == 3:
            recon, indices, commit = out
        else:
            recon, indices = out
            commit = motion.new_zeros(())

        recon_loss = reconstruction_loss(recon, motion)
        total = recon_loss + self.commit_beta * commit

        self.opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.tokenizer.parameters(), self.grad_clip)
        self.opt.step()
        self.ema.update(self.tokenizer)

        return {
            "recon": recon_loss.item(),
            "commit": commit.item(),
            "total": total.item(),
            "perplexity": self._perplexity(indices),
        }
