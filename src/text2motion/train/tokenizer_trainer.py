"""Tokenizer trainer (Contribution A) -- trains the Residual-FSQ tokenizer or the strong-RVQ baseline.

One step: encode->quantize->decode a batch of normalized motion windows and minimise the reconstruction
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

    def _code_stats(self, indices: torch.Tensor) -> tuple[float, float]:
        """Codebook-health signals, averaged over codebooks:
        - perplexity = exp(entropy of code usage) = EFFECTIVE codes used this batch;
        - usage_frac = fraction of the codebook actually hit (alive). Low usage = collapse (the thing
          RVQ fights with dead-code reset; for FSQ unused grid points are free, so it only needs to be
          'high enough'). Same metric, read per quantizer."""
        # bincount is non-deterministic on CUDA (atomics) -> raises under
        # use_deterministic_algorithms(True). This is a diagnostic, not in the training graph, so do
        # it on CPU (deterministic, cheap for small int index tensors).
        indices = indices.detach().cpu()
        perplexities, usages = [], []
        for codebook in range(indices.shape[-1]):
            counts = torch.bincount(
                indices[..., codebook].reshape(-1), minlength=self.codebook_size
            ).float()
            probs = counts[counts > 0] / counts.sum()
            perplexities.append(float(torch.exp(-(probs * probs.log()).sum())))
            usages.append(float((counts > 0).float().mean()))
        n = len(perplexities)
        return sum(perplexities) / n, sum(usages) / n

    def train_step(self, motion: torch.Tensor) -> dict[str, float]:
        """motion (B, window, 263) normalized. Returns two tiers of scalars: COMMON (recon, total)
        shared by both tokenizers, and HEALTH (quantizer-specific: perplexity, usage_frac, commit)
        that justifies a healthy tokenizer from either the RVQ or the FSQ point of view."""
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

        perplexity, usage_frac = self._code_stats(indices)
        return {
            # --- common (both quantizers) ---
            "recon": recon_loss.item(),
            "total": total.item(),
            # --- quantizer health ---
            "perplexity": perplexity,
            "usage_frac": usage_frac,
            "commit": commit.item(),  # RVQ machinery; structurally 0 for FSQ (no codebook to commit)
        }
