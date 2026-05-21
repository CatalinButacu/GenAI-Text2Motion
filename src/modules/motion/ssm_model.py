from __future__ import annotations

import logging
import os
from dataclasses import fields as dc_fields

import torch
import torch.nn.functional as F

from src.modules.motion.config import TrainingConfig
from src.modules.motion.nn_models import TextToMotionSSM
from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer
from src.shared.tokenizer import tokenize
from src.utils.mem_profile import profile_memory

from .models import MotionClip, MotionSource

log = logging.getLogger(__name__)


def sample_indices(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample token indices from (B, T', K, V) logits.

    temperature=1.0 and top_p=1.0 is identical to argmax (greedy/deterministic).
    temperature>1 diversifies output; top_p<1 applies nucleus (top-p) filtering.
    """
    if temperature <= 0.0 or (temperature == 1.0 and top_p >= 1.0):
        return logits.argmax(dim=-1)

    B, T, K, V = logits.shape
    scaled = logits / temperature

    if top_p < 1.0:
        probs = F.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumprobs = sorted_probs.cumsum(dim=-1)
        # Zero out tokens beyond the top-p nucleus (keep at least 1 token per position)
        remove = (cumprobs - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs.scatter_(-1, sorted_idx, sorted_probs)
        flat = probs.reshape(B * T * K, V)
    else:
        flat = F.softmax(scaled.reshape(B * T * K, V), dim=-1)

    indices = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return indices.reshape(B, T, K)


def sample_one_codebook(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample one (B, T, V) logits tensor into (B, T) indices, with the same
    temperature / top-p semantics as sample_indices."""
    if temperature <= 0.0 or (temperature == 1.0 and top_p >= 1.0):
        return logits.argmax(dim=-1)
    B, T, V = logits.shape
    scaled = logits / temperature

    if top_p < 1.0:
        probs = F.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumprobs = sorted_probs.cumsum(dim=-1)
        remove = (cumprobs - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs.scatter_(-1, sorted_idx, sorted_probs)
        flat = probs.reshape(B * T, V)
    else:
        flat = F.softmax(scaled.reshape(B * T, V), dim=-1)
    idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return idx.reshape(B, T)


def sample_ar_k(
    model,
    inputs,
    motion_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    cfg_scale: float = 1.0,
    uncond_inputs=None,
) -> torch.Tensor:
    """Autoregressive sampling across K codebooks for a model with the
    ResidualKHead arch.

    For each latent frame, samples codebook 0, embeds the chosen token via
    decoder.token_embeds[0], adds it to the residual conditioning signal,
    then samples codebook 1, and so on. This matches the RVQ residual
    structure that the head was trained against (vs. naive independent
    sampling across K).

    Returns: (B, T', K) codebook indices ready for tokenizer.decode.

    cfg_scale > 1.0 + uncond_inputs not None applies CFG to each codebook's
    logits independently: guided = uncond + cfg * (cond - uncond).
    """
    use_cfg = cfg_scale > 1.0 and uncond_inputs is not None
    features, _ = model.forward_features(inputs, motion_length)
    feat_unc = None

    if use_cfg:
        feat_unc, _ = model.forward_features(uncond_inputs, motion_length)
    head = model.decoder  # ResidualKHead
    K = head.n_codebooks
    running = torch.zeros_like(features)
    running_unc = torch.zeros_like(feat_unc) if use_cfg else None
    all_tokens: list[torch.Tensor] = []

    for k in range(K):
        logits_k = head.head_for_codebook(features, running, k)

        if use_cfg:
            assert feat_unc is not None and running_unc is not None
            uncond_logits_k = head.head_for_codebook(feat_unc, running_unc, k)
            logits_k = uncond_logits_k + cfg_scale * (logits_k - uncond_logits_k)
        tok_k = sample_one_codebook(logits_k, temperature, top_p)  # (B, T')
        all_tokens.append(tok_k)

        if k < K - 1:
            running = running + head.embed_token(tok_k, k)

            if use_cfg:
                # Use the same chosen token for the uncond running embed.
                # The uncond pathway sees what the conditional path committed
                # to -- this is the standard CFG-with-AR formulation.
                running_unc = running_unc + head.embed_token(tok_k, k)  # type: ignore[operator]
    return torch.stack(all_tokens, dim=-1)  # (B, T', K)


def coerce_config(raw) -> TrainingConfig:
    if isinstance(raw, TrainingConfig):
        return raw
    field_vals = {f.name: getattr(raw, f.name) for f in dc_fields(raw)}
    return TrainingConfig(**field_vals)


class SSMMotionModel:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/motion_ssm/best_model.pt",
        rvq_checkpoint_path: str = "checkpoints/rvq_tokenizer/best_model.pt",
        data_dir: str = "data/AMASS",
    ) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"[SSM] checkpoint not found: {checkpoint_path!r}. "
                "Train one via `python scripts/training/train_motion_ssm.py`."
            )
        if not os.path.exists(rvq_checkpoint_path):
            raise FileNotFoundError(
                f"[SSM] RVQ tokenizer not found: {rvq_checkpoint_path!r}. "
                "Train one via `python scripts/training/train_rvq_tokenizer.py`."
            )

        self.checkpoint_path = checkpoint_path
        self.rvq_checkpoint_path = rvq_checkpoint_path
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ck = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.vocab: dict = ck["vocab"]
        cfg = coerce_config(ck["config"])
        self.model = TextToMotionSSM(cfg).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

        rvq_ck = torch.load(rvq_checkpoint_path, map_location=self.device, weights_only=False)
        self.tokenizer = MotionRVQTokenizer(
            motion_dim=cfg.motion_dim,
            latent_dim=cfg.rvq_latent_dim,
            n_codebooks=cfg.rvq_n_codebooks,
            codebook_size=cfg.rvq_codebook_size,
            down_t=cfg.rvq_down_t,
        ).to(self.device)
        self.tokenizer.load_state_dict(rvq_ck["model_state_dict"])
        self.tokenizer.eval()

        log.info(
            "[SSM] loaded %s (val_loss=%s) + rvq %s (val_loss=%s)",
            checkpoint_path,
            ck.get("val_loss", ck.get("val_loss", "N/A")),
            rvq_checkpoint_path,
            rvq_ck.get("val_loss", "N/A"),
        )

    @profile_memory
    def generate_from_text_tokens(
        self,
        text: str,
        num_frames: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        cfg_scale: float = 1.0,
    ) -> MotionClip:
        """Sample motion from text.

        cfg_scale > 1.0 enables Classifier-Free Guidance: blend conditional and
        unconditional logits as `unc + cfg_scale * (cond - unc)`. Strengthens
        text adherence at the cost of diversity. Requires use_sbert=True (the
        encoder needs to embed the empty prompt) AND the model was trained
        with cfg_dropout_prob > 0 so the empty path is meaningful.
        """
        use_sbert = getattr(self.model.config, "use_sbert", False)
        # CFG only meaningful for SBERT/CLIP-conditioned models. For the
        # legacy token-id path, fall back to vanilla sampling.
        use_cfg = cfg_scale > 1.0 and use_sbert
        uncond_inputs: list[str] | None = None

        if use_sbert:
            inputs = [text]
            if use_cfg:
                uncond_inputs = [""]
        else:
            inputs = (
                torch.tensor(tokenize(text, self.vocab), dtype=torch.long)
                .unsqueeze(0)
                .to(self.device)
            )

        # AR head models sample sequentially across K codebooks; independent
        # head models sample all K positions in parallel via sample_indices.
        arch = getattr(self.model, "arch", getattr(self.model.config, "arch", "independent"))
        use_ar = arch == "residual_k"

        with torch.no_grad():
            if use_ar:
                indices = sample_ar_k(
                    self.model, inputs, num_frames,
                    temperature=temperature, top_p=top_p,
                    cfg_scale=cfg_scale if use_cfg else 1.0,
                    uncond_inputs=uncond_inputs if use_cfg else None,
                )
            else:
                logits, _ = self.model(inputs, num_frames)  # (B, T', K, V)

                if use_cfg and uncond_inputs is not None:
                    uncond_logits, _ = self.model(uncond_inputs, num_frames)
                    # guided = uncond + scale * (cond - uncond)
                    logits = uncond_logits + cfg_scale * (logits - uncond_logits)
                indices = sample_indices(logits, temperature, top_p)  # (B, T', K)
            motion = self.tokenizer.decode(indices)  # (B, T, motion_dim)
            motion = motion[:, :num_frames]  # trim to requested length
            motion = motion.cpu().numpy()[0]

        return MotionClip(action=text, smplx_params=motion, source=MotionSource.SSM)

    def invoke(self, text: str, duration_s: float = 3.0) -> MotionClip:
        return self.generate_from_text_tokens(text, int(duration_s * 30))
