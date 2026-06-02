from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import fields as dc_fields

import numpy as np
import torch
import torch.nn.functional as F

from src.architecture.nn_models import TextToMotionSSM
from src.architecture.rvq_tokenizer import MotionRVQTokenizer
from src.architecture.training.trainer_utils import load_compatible, load_strict
from src.modules.runtime import StreamCapabilityError
from src.shared.config import TrainingConfig
from src.shared.tokenizer import tokenize
from src.utils.mem_profile import profile_memory

from .models import MotionClip, MotionSource

log = logging.getLogger(__name__)


def detect_coord_system(cfg) -> str:
    sources = getattr(cfg, "unified_sources", None) or getattr(cfg, "unifiedSources", None)

    if sources:
        return "yup" if list(sources) == ["humanml3d"] else "zup"

    data_dir = (getattr(cfg, "data_dir", "") or "").lower()
    return "yup" if "humanml3d" in data_dir else "zup"


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """In-place nucleus filter: zero out tokens beyond the top-p mass per row."""
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumprobs = sorted_probs.cumsum(dim=-1)
    remove = (cumprobs - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    probs.scatter_(-1, sorted_idx, sorted_probs)

    return probs


def sample_indices(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample token indices from (B, T', K, V) logits. T=1, top_p=1 is greedy argmax."""
    if temperature <= 0.0 or (abs(temperature - 1.0) < 1e-9 and top_p >= 1.0):
        return logits.argmax(dim=-1)

    B, T, K, V = logits.shape
    scaled = logits / temperature

    if top_p < 1.0:
        probs = apply_top_p(F.softmax(scaled, dim=-1), top_p)
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
    """Sample one (B, T, V) logits tensor into (B, T) indices with same T/top-p semantics."""
    if temperature <= 0.0 or (abs(temperature - 1.0) < 1e-9 and top_p >= 1.0):
        return logits.argmax(dim=-1)
    B, T, V = logits.shape
    scaled = logits / temperature

    if top_p < 1.0:
        probs = apply_top_p(F.softmax(scaled, dim=-1), top_p)
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
    """AR sampling across K codebooks (ResidualKHead arch). CFG via uncond_inputs + cfg_scale>1.
    Returns (B, T', K) codebook indices ready for tokenizer.decode."""
    use_cfg = cfg_scale > 1.0 and uncond_inputs is not None
    features, _ = model.forward_features(inputs, motion_length)
    feat_unc = None

    if use_cfg:
        feat_unc, _ = model.forward_features(uncond_inputs, motion_length)
    head = model.decoder  # ResidualKHead
    K = head.n_codebooks
    running = torch.zeros_like(features)
    running_unc = torch.zeros_like(feat_unc) if feat_unc is not None else None
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
                # Standard CFG-with-AR: uncond pathway sees the chosen conditional token.
                running_unc = running_unc + head.embed_token(tok_k, k)  # type: ignore[operator]
    return torch.stack(all_tokens, dim=-1)  # (B, T', K)


def snake_to_camel(name: str) -> str:
    """snake_case -> snakeCase (for legacy checkpoint compat)."""
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def coerce_config(raw) -> TrainingConfig:
    """Restore TrainingConfig from checkpoint; recovers camelCase keys from legacy pickles."""
    raw_dict = vars(raw) if hasattr(raw, "__dict__") else dict(raw)

    if not raw_dict and isinstance(raw, TrainingConfig):
        # Freshly-constructed TrainingConfig with no overrides: return as-is.
        return raw
    field_vals: dict = {}

    for f in dc_fields(TrainingConfig):
        if f.name in raw_dict:
            field_vals[f.name] = raw_dict[f.name]
            continue
        legacy = snake_to_camel(f.name)

        if legacy in raw_dict:
            field_vals[f.name] = raw_dict[legacy]
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
                f"checkpoint not found: {checkpoint_path!r}. "
                "Train one via `python scripts/training/train_motion_ssm.py`."
            )
        if not os.path.exists(rvq_checkpoint_path):
            raise FileNotFoundError(
                f"RVQ tokenizer not found: {rvq_checkpoint_path!r}. "
                "Train one via `python scripts/training/train_rvq_tokenizer.py`."
            )

        self.checkpoint_path = checkpoint_path
        self.rvq_checkpoint_path = rvq_checkpoint_path
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ck = torch.load(  # NOSONAR – ckpt has Python dataclass; weights_only=True not viable
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.vocab: dict = ck["vocab"]
        cfg = coerce_config(ck["config"])
        self.coord_system = detect_coord_system(cfg)
        self.model = TextToMotionSSM(cfg).to(self.device)
        # SSM_STRICT_LOAD=1 enforces strict matching; default is lenient-with-warning.
        strict = os.environ.get("SSM_STRICT_LOAD", "").lower() in ("1", "true", "yes")

        if strict:
            load_strict(self.model, ck["model_state_dict"], "ssm_model")
        else:
            load_compatible(self.model, ck["model_state_dict"], "ssm_model")
        self.model.eval()

        rvq_ck = torch.load(  # NOSONAR – ckpt has Python dataclass; weights_only=True not viable
            rvq_checkpoint_path, map_location=self.device, weights_only=False
        )
        self.tokenizer = MotionRVQTokenizer(
            motion_dim=cfg.motion_dim,
            latent_dim=cfg.rvq_latent_dim,
            n_codebooks=cfg.rvq_n_codebooks,
            codebook_size=cfg.rvq_codebook_size,
            down_t=cfg.rvq_down_t,
        ).to(self.device)

        if strict:
            load_strict(self.tokenizer, rvq_ck["model_state_dict"], "rvq_tokenizer")
        else:
            load_compatible(self.tokenizer, rvq_ck["model_state_dict"], "rvq_tokenizer")
        self.tokenizer.eval()
        # Flag baked in at training time; fall back to inferring from model config.
        self.streaming_capable: bool = ck.get("streaming_capable", not cfg.bidirectional)

        log.info(
            "loaded %s (val_loss=%s) + rvq %s (val_loss=%s)",
            checkpoint_path,
            ck.get("val_loss", "N/A"),
            rvq_checkpoint_path,
            rvq_ck.get("val_loss", "N/A"),
        )

    def is_stream_capable(self) -> bool:
        return self.streaming_capable

    def require_stream_capable(self) -> None:
        if not self.is_stream_capable():
            raise StreamCapabilityError(
                "checkpoint was trained with bidirectional=True; "
                "streaming requires bidirectional=False -- "
                "retrain with --bidirectional disabled or supply a causal checkpoint"
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
        """Sample motion from text. cfg_scale>1 enables CFG (requires use_sbert + cfg_dropout)."""
        use_sbert = getattr(self.model.config, "use_sbert", False)
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

        arch = getattr(self.model, "arch", getattr(self.model.config, "arch", "independent"))
        use_ar = arch == "residual_k"

        with torch.no_grad():
            if use_ar:
                indices = sample_ar_k(
                    self.model,
                    inputs,
                    num_frames,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_scale=cfg_scale if use_cfg else 1.0,
                    uncond_inputs=uncond_inputs if use_cfg else None,
                )
            else:
                logits, _ = self.model(inputs, num_frames)  # (B, T', K, V)

                if use_cfg and uncond_inputs is not None:
                    uncond_logits, _ = self.model(uncond_inputs, num_frames)
                    logits = uncond_logits + cfg_scale * (logits - uncond_logits)
                indices = sample_indices(logits, temperature, top_p)  # (B, T', K)
            motion = self.tokenizer.decode(indices)  # (B, T, motion_dim)
            motion = motion[:, :num_frames]  # trim to requested length
            motion = motion.cpu().numpy()[0]

        return MotionClip(
            action=text,
            smplx_params=motion,
            source=MotionSource.SSM,
            coord_system=self.coord_system,
        )

    def prepare_inputs(self, text: str) -> list[str] | torch.Tensor:
        if getattr(self.model.config, "use_sbert", False):
            return [text]

        return (
            torch.tensor(tokenize(text, self.vocab), dtype=torch.long).unsqueeze(0).to(self.device)
        )

    def emit_chunks(
        self,
        state,
        num_frames: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[np.ndarray]:
        rvq_down_t = self.model.config.rvq_down_t
        latent_steps_needed = (num_frames + rvq_down_t - 1) // rvq_down_t
        steps_left = state.max_steps - state.latent_step
        steps = min(latent_steps_needed, steps_left)
        frames_emitted = 0

        for _ in range(steps):
            logits, _, state = self.model.stream_step(state)
            indices = sample_indices(logits, temperature, top_p)
            chunk = self.tokenizer.decode(indices).cpu().numpy()[0]
            remaining = num_frames - frames_emitted

            if remaining < len(chunk):
                chunk = chunk[:remaining]

            frames_emitted += len(chunk)
            yield chunk

            if frames_emitted >= num_frames:
                break

    def stream_actions(
        self,
        actions: list[tuple[str, int]],
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Iterator[np.ndarray]:
        state = None

        with torch.no_grad():
            for text, num_frames in actions:
                inputs = self.prepare_inputs(text)

                if state is None:
                    state = self.model.stream_begin(inputs)
                else:
                    new_cond = self.model.condition_proj(self.model.text_encoder(inputs))
                    state.carry_over(new_cond)

                yield from self.emit_chunks(state, num_frames, temperature, top_p)
