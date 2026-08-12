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
        recon, indices, commit = self.tokenizer(motion)

        recon_loss = reconstruction_loss(recon, motion)
        total = recon_loss + self.commit_beta * commit

        self.opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.tokenizer.parameters(), self.grad_clip)
        self.opt.step()
        self.ema.update(self.tokenizer)

        perplexity, usage_frac = self._code_stats(indices)
        return {
            "recon": recon_loss.item(),
            "total": total.item(),
            "perplexity": perplexity,
            "usage_frac": usage_frac,
            "commit": commit.item(),  # RVQ machinery; structurally 0 for FSQ (no codebook to commit)
        }
