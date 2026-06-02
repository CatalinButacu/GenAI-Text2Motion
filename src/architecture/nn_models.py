from __future__ import annotations

import logging

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.architecture.ssm import BiMambaLayer, MambaLayer, SSMConfig
from src.architecture.streaming import StreamingState

log = logging.getLogger(__name__)

EMBEDDING_INIT_STD = 0.02


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_length: int):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.register_buffer("pos_ids", torch.arange(max_length).unsqueeze(0))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        pos = self.pos_ids[:, :s].expand(b, -1)  # type: ignore[index]
        x = self.word_embed(token_ids) + self.pos_embed(pos)

        return self.encoder(x).mean(dim=1)


class PretrainedTextEncoder(nn.Module):
    """SBERT/CLIP wrapper that projects sentence embeddings to d_model; frozen by default."""

    def __init__(self, d_model: int, model_name: str = "all-MiniLM-L6-v2", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.sbert = SentenceTransformer(model_name)

        if freeze:
            for param in self.sbert.parameters():
                param.requires_grad = False

        # get_sentence_embedding_dimension() returns None for CLIP; probe with one-token encode.
        with torch.no_grad():
            probe = self.sbert.encode(["x"], convert_to_tensor=True, show_progress_bar=False)
            self.encoder_dim = int(probe.shape[-1])
        log.info(
            "PretrainedTextEncoder: %s (dim=%d, frozen=%s)",
            model_name,
            self.encoder_dim,
            freeze,
        )
        self.proj = nn.Sequential(nn.Linear(self.encoder_dim, d_model), nn.LayerNorm(d_model))

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.proj.parameters()).device

        with torch.no_grad():
            emb = (
                self.sbert.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .to(device)
                .clone()
            )

        return self.proj(emb)


# Backward-compat alias for old checkpoints/code; new code uses PretrainedTextEncoder.
SBERTTextEncoder = PretrainedTextEncoder


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: norm(x)*gamma + beta with text-conditioned gamma/beta.
    Identity init (gamma=1, beta=0) at start of training."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_cond, d_model * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:d_model])
        nn.init.zeros_(self.proj.bias[d_model:])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)

        return gamma * self.norm(x) + beta


class RVQMotionDecoder(nn.Module):
    """Discrete RVQ token head: per-latent-frame K independent classifiers over V codebook entries.
    Ignores RVQ residual structure (k>0 codebooks see only SSM features, not prior codebook tokens).
    Use ResidualKHead for the residual-aware variant; pick via ModelConfig.arch."""

    def __init__(
        self,
        d_model: int,
        n_codebooks: int,
        codebook_size: int,
        max_length: int,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.max_length = max_length
        self.length_head = nn.Linear(d_model, 1)
        self.token_heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size) for _ in range(n_codebooks)]
        )

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
        target_tokens: torch.Tensor | None = None,  # noqa: ARG002 -- matches ResidualKHead signature
    ) -> tuple:
        logits = torch.stack([head(features) for head in self.token_heads], dim=2)
        length_pred = torch.sigmoid(self.length_head(condition)).squeeze(-1) * self.max_length

        return logits, length_pred


class ResidualKHead(nn.Module):
    """Autoregressive K-codebook head: each codebook sees features + sum of embedded prior tokens.
    Training (target_tokens given): teacher-forced sequential rollout across K codebooks.
    Inference: use sample_ar_k() in ssm_model.py for proper sampling (temperature/top-p/CFG).
    Calling forward() at inference does argmax-self-feed -- a debug path only.
    Requires retraining; legacy RVQMotionDecoder weights don't map onto this layer set."""

    def __init__(
        self,
        d_model: int,
        n_codebooks: int,
        codebook_size: int,
        max_length: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.max_length = max_length
        self.length_head = nn.Linear(d_model, 1)
        self.token_heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size) for _ in range(n_codebooks)]
        )
        # Per-codebook embedding back to d_model; summed cumulatively into conditioning.
        self.token_embeds = nn.ModuleList(
            [nn.Embedding(codebook_size, d_model) for _ in range(n_codebooks)]
        )
        # Small init so AR head matches independent head at step 0 (drifts via gradients after).

        for emb in self.token_embeds:
            nn.init.normal_(emb.weight, mean=0.0, std=EMBEDDING_INIT_STD)

    def head_for_codebook(
        self,
        features: torch.Tensor,
        running_embed: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Per-codebook logits; exposed for inference sampler's per-codebook rollout."""
        return self.token_heads[k](features + running_embed)

    def embed_token(self, token: torch.Tensor, k: int) -> torch.Tensor:
        return self.token_embeds[k](token)

    def length_pred(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.length_head(condition)).squeeze(-1) * self.max_length

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
    ) -> tuple:
        running_embed = torch.zeros_like(features)
        all_logits: list[torch.Tensor] = []

        for k in range(self.n_codebooks):
            logits_k = self.head_for_codebook(features, running_embed, k)
            all_logits.append(logits_k)

            if k < self.n_codebooks - 1:
                if target_tokens is not None:
                    tok = target_tokens[:, :, k]
                else:
                    tok = logits_k.argmax(dim=-1)
                running_embed = running_embed + self.embed_token(tok, k)
        logits = torch.stack(all_logits, dim=2)  # (B, T', K, V)

        return logits, self.length_pred(condition)


class TextToMotionSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.use_sbert:
            self.text_encoder = PretrainedTextEncoder(
                d_model=config.d_model,
                model_name=config.sbert_model,
                freeze=config.freeze_sbert,
            )
            self.condition_proj = nn.Identity()
        else:
            self.text_encoder = SimpleTextEncoder(
                config.vocab_size,
                config.text_embed_dim,
                config.max_text_length,
            )
            self.condition_proj = nn.Linear(config.text_embed_dim, config.d_model)

        self.latent_length = config.max_motion_length // config.rvq_down_t
        self.pos_embed = nn.Embedding(self.latent_length, config.d_model)
        self.register_buffer("motion_pos_ids", torch.arange(self.latent_length))

        bidirectional = config.bidirectional
        grad_ckpt = config.gradient_checkpointing
        use_film = config.use_film
        layer_cls = BiMambaLayer if bidirectional else MambaLayer
        ssm_cfg = SSMConfig(
            d_model=config.d_model, d_state=config.d_state, gradient_checkpointing=grad_ckpt
        )
        self.layers = nn.ModuleList([layer_cls(ssm_cfg) for _ in range(config.n_layers)])
        self.dropout = nn.Dropout(getattr(config, "model_dropout", 0.0))
        self.use_film = use_film
        self.bidirectional = bidirectional
        self.ssm_cfg = ssm_cfg

        if use_film:
            self.films = nn.ModuleList(
                [FiLM(config.d_model, config.d_model) for _ in range(config.n_layers)]
            )
        else:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(config.d_model) for _ in range(config.n_layers)]
            )
        arch = getattr(config, "arch", "independent")
        decoder_cls = ResidualKHead if arch == "residual_k" else RVQMotionDecoder
        self.decoder = decoder_cls(
            d_model=config.d_model,
            n_codebooks=config.rvq_n_codebooks,
            codebook_size=config.rvq_codebook_size,
            max_length=config.max_motion_length,
        )
        self.arch = arch
        # Pose-prefix curriculum: projects rvq_latent_dim seed -> d_model. Unused when prob=0.
        self.seed_projector = nn.Linear(config.rvq_latent_dim, config.d_model)

    def seed_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Project tokenizer latent (B, P, rvq_latent_dim) -> (B, P, d_model) for pose-prefix."""
        return self.seed_projector(latent)

    def forward_features(
        self,
        inputs: torch.Tensor | list[str],
        motion_length: int | None = None,
        seed_latent: torch.Tensor | None = None,
    ) -> tuple:
        """SSM trunk only -> (features, cond). seed_latent (B, P', d_model) prepends prefix
        latents to advance the recurrent state, then drops them to keep output shape unchanged."""
        cond = self.condition_proj(self.text_encoder(inputs))

        if motion_length is None or motion_length > self.config.max_motion_length:
            motion_length = self.config.max_motion_length
        assert isinstance(motion_length, int)
        latent_len = max(1, motion_length // self.config.rvq_down_t)
        seed_len = 0 if seed_latent is None else seed_latent.shape[1]
        total_len = seed_len + latent_len

        if total_len > self.latent_length:
            raise ValueError(
                f"seed_latent ({seed_len}) + prediction ({latent_len}) exceeds "
                f"max latent length {self.latent_length}; lower motion_length "
                "or shorten the seed prefix"
            )
        pos = self.pos_embed(self.motion_pos_ids[:total_len])  # type: ignore[index]
        x_pred = cond.unsqueeze(1) + pos[seed_len:].unsqueeze(0)  # (B, latent_len, d_model)

        if seed_latent is not None:
            x_seed = seed_latent + pos[:seed_len].unsqueeze(0)  # (B, seed_len, d_model)
            x = torch.cat([x_seed, x_pred], dim=1)  # (B, total_len, d_model)
        else:
            x = x_pred

        if self.use_film:
            for layer, film in zip(self.layers, self.films):
                x = x + self.dropout(layer(film(x, cond)))
        else:
            for layer, norm in zip(self.layers, self.norms):
                x = x + self.dropout(layer(norm(x)))

        # Drop seed prefix from output so prediction shape matches the no-seed path.
        if seed_latent is not None:
            x = x[:, seed_len:]

        return x, cond

    def forward(
        self,
        inputs: torch.Tensor | list[str],
        motion_length: int | None = None,
        target_tokens: torch.Tensor | None = None,
        seed_latent: torch.Tensor | None = None,
    ) -> tuple:
        """Parallel forward over latent (downsampled) frames.
        Returns (logits (B, T', K, V), length_pred (B,))."""
        x, cond = self.forward_features(inputs, motion_length, seed_latent=seed_latent)
        return self.decoder(x, cond, target_tokens=target_tokens)

    def stream_begin(self, inputs: torch.Tensor | list[str]) -> StreamingState:
        """Initialise streaming state for a new action; encodes text, zero-inits SSM hidden states.
        For action transitions, call StreamingState.carry_over() on the returned state instead."""
        if self.bidirectional:
            raise RuntimeError(
                "stream_begin() requires config.bidirectional=False; "
                "BiMambaLayer has no causal single-step inference path"
            )
        cond = self.condition_proj(self.text_encoder(inputs))
        d_inner = self.ssm_cfg.d_inner
        d_state = self.ssm_cfg.d_state
        batch = cond.shape[0]
        device = cond.device
        dtype = cond.dtype
        h0 = [torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype) for _ in self.layers]

        return StreamingState(
            cond=cond,
            layer_h=h0,
            layer_conv=[None] * len(self.layers),
            latent_step=0,
            max_steps=self.latent_length,
        )

    def stream_step(
        self, state: StreamingState, target_tokens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, StreamingState]:
        """Emit ONE latent step. Returns (logits (B, 1, K, V), length_pred (B,), state).
        Mutates state in place: increments latent_step and advances per-layer SSM state."""
        t = state.latent_step

        if t >= state.max_steps:
            raise RuntimeError(
                f"stream_step() past max_steps={state.max_steps}; "
                "begin a new action via carry_over() or stream_begin()"
            )

        if len(state.layer_h) != len(self.layers):
            raise RuntimeError(
                f"stream_step(): state has {len(state.layer_h)} layer hidden "
                f"states but model has {len(self.layers)} layers -- the state "
                "was probably built from a different model. Re-init via "
                "stream_begin() on THIS model."
            )
        pos_id = self.motion_pos_ids[t : t + 1]  # type: ignore[index]
        pos = self.pos_embed(pos_id)  # (1, d_model)
        x = state.cond + pos  # initial latent: cond + positional embedding

        for i, layer in enumerate(self.layers):
            if not isinstance(layer, MambaLayer):
                raise RuntimeError(
                    f"stream_step requires MambaLayer; got {type(layer).__name__} at layer {i}"
                )

            if self.use_film:
                x_norm = self.films[i](x.unsqueeze(1), state.cond).squeeze(1)
            else:
                x_norm = self.norms[i](x)
            out_t, h_new, conv_buf_new = layer.step(x_norm, state.layer_h[i], state.layer_conv[i])
            state.layer_h[i] = h_new
            state.layer_conv[i] = conv_buf_new
            x = x + out_t  # residual matches parallel forward

        # Decoder expects (B, T'=1, d_model); independent head ignores target_tokens.
        logits, length_pred = self.decoder(
            x.unsqueeze(1),
            state.cond,
            target_tokens=target_tokens,
        )
        state.latent_step = t + 1

        return logits, length_pred, state
