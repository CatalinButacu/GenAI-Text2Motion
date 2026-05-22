from __future__ import annotations

import logging

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.modules.motion.ssm import BiMambaLayer, MambaLayer, SSMConfig
from src.modules.motion.streaming import StreamingState

log = logging.getLogger(__name__)


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
    """Wraps any sentence-transformers model and projects its output to d_model.

    Works with SBERT variants (all-MiniLM-L6-v2 -> 384-d, all-mpnet-base-v2 -> 768-d)
    AND CLIP text encoders (clip-ViT-B-32 -> 512-d, clip-ViT-L-14 -> 768-d).

    The encoder is loaded by `model_name`; output dim is probed at construction
    rather than hardcoded. Frozen by default -- the motion training treats the
    pretrained semantic prior as a fixed feature extractor.
    """

    def __init__(self, d_model: int, model_name: str = "all-MiniLM-L6-v2", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.sbert = SentenceTransformer(model_name)

        if freeze:
            for param in self.sbert.parameters():
                param.requires_grad = False
        # Probe the actual embedding dim. sentence-transformers'
        # get_sentence_embedding_dimension() returns None for CLIP wrappers,
        # so encode a one-token string and read the output shape instead.
        with torch.no_grad():
            probe = self.sbert.encode(["x"], convert_to_tensor=True, show_progress_bar=False)
            self.encoder_dim = int(probe.shape[-1])
        self.available = True
        log.info(
            "PretrainedTextEncoder: %s (dim=%d, frozen=%s)",
            model_name, self.encoder_dim, freeze,
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


# Backward-compat alias. Old checkpoints and code that imported the SBERT-specific
# name still work; new code should use PretrainedTextEncoder.
SBERTTextEncoder = PretrainedTextEncoder


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: text-conditioned scale+shift at every SSM layer.

    Replaces plain LayerNorm with norm(x)*gamma + beta where gamma,beta are
    predicted from the text condition vector. Initialised so gamma=1, beta=0
    (identity at start of training).
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_cond, d_model * 2)
        # Identity init: gamma->1, beta->0
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:d_model])
        nn.init.zeros_(self.proj.bias[d_model:])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)  cond: (B, d_cond)
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)

        return gamma * self.norm(x) + beta


class RVQMotionDecoder(nn.Module):
    """Discrete RVQ token head: predicts K codebook indices per latent frame.

    The SSM trunk emits (B, T', d_model) features, where T' = max_motion_length // down_t.
    Per latent frame, one classifier per codebook emits logits over V entries.

    Limitation: the K classifiers are independent given the features. RVQ is a
    *residual* code (codebook k quantizes the residual after codebooks 0..k-1),
    but this head ignores that structure at both train and inference time. The
    ResidualKHead below restores it; pick which one via ModelConfig.arch.
    """

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
        target_tokens: torch.Tensor | None = None,  # noqa: ARG002 -- ignored; matches ResidualKHead signature
    ) -> tuple:
        # features: (B, T', d_model), condition: (B, d_model)
        # returns: logits (B, T', K, V), length_pred (B,)
        logits = torch.stack([head(features) for head in self.token_heads], dim=2)
        length_pred = torch.sigmoid(self.length_head(condition)).squeeze(-1) * self.max_length

        return logits, length_pred


class ResidualKHead(nn.Module):
    """Autoregressive K-codebook head: each codebook's classifier sees the SSM
    features PLUS the sum of embedded prior-codebook tokens.

    Training (when target_tokens is passed):
        codebook 0:  logits_0 = head_0(features)
                     -> sample/argmax/teacher-force token t_0
                     -> running = embed_0(t_0)
        codebook k:  logits_k = head_k(features + running)
                     -> running = running + embed_k(t_k)
        Output: stacked logits (B, T', K, V)

    Inference:
        Use sample_ar_k() in ssm_model.py for proper sequential sampling
        with temperature / top-p / CFG support. Calling forward() without
        target_tokens at inference does argmax-self-feed, which is a
        debug/reference path -- temperature won't take effect.

    Param overhead vs RVQMotionDecoder: ~K * codebook_size * d_model extra
    (~1M params at K=6, V=512, d_model=384). Tiny relative to the SSM trunk.

    Restoring the residual structure is the only known architectural defect
    in the previous head; expected ~0.1-0.3 nat gain on token CE per the
    2026-05 technique audit. Requires retraining -- the legacy
    RVQMotionDecoder weights cannot map onto this layer set.
    """

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
        # One embedding per codebook to project its chosen token back into
        # d_model space; summed cumulatively into the conditioning signal.
        self.token_embeds = nn.ModuleList(
            [nn.Embedding(codebook_size, d_model) for _ in range(n_codebooks)]
        )
        # Initialise embeddings small so codebook-0's prediction is unchanged
        # at the start of training (the AR head behaves like the independent
        # head until gradients accumulate).
        for emb in self.token_embeds:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def head_for_codebook(
        self, features: torch.Tensor, running_embed: torch.Tensor, k: int,
    ) -> torch.Tensor:
        """Per-codebook logits computation, exposed so the inference sampler
        can call it one codebook at a time."""
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
        # features: (B, T', d_model), condition: (B, d_model)
        # target_tokens: (B, T', K) or None (teacher forcing vs greedy self-feed)
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
        # Projects tokenizer latent (rvq_latent_dim) up to d_model so the
        # trainer-side pose-prefix curriculum can feed prior-action context
        # as a seed_latent. Untrained / unused when pose_prefix_prob = 0.
        self.seed_projector = nn.Linear(config.rvq_latent_dim, config.d_model)

    def seed_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Project tokenizer-encoded latent (B, P, rvq_latent_dim) up to (B, P, d_model).

        Used by the trainer-side pose-prefix curriculum: encode the prefix of
        a motion clip through the frozen tokenizer's RVQ decode path to get
        (B, P, rvq_latent_dim), then call this method to produce a seed_latent
        suitable for :meth:`forward`. The projection is part of the model's
        parameter set so it co-trains with the SSM trunk.
        """
        return self.seed_projector(latent)

    def forward_features(
        self,
        inputs: torch.Tensor | list[str],
        motion_length: int | None = None,
        seed_latent: torch.Tensor | None = None,
    ) -> tuple:
        """Run the SSM trunk only, returning (features, cond) without the
        decoder head. Used by the AR sampler so it can call
        decoder.head_for_codebook one codebook at a time.

        ``seed_latent``: optional ``(B, P', d_model)`` pose-prefix conditioning.
        When provided, those P' latent steps are prepended to the SSM input so
        the recurrent state advances through them before producing the new
        prediction. The seed portion is *dropped* from the returned features,
        so the output shape is unchanged from the no-seed path. This is the
        training-time pose-prefix curriculum hook and the offline analogue of
        :meth:`stream_step`'s hidden-state carryover.
        """
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
                x = x + layer(film(x, cond))
        else:
            for layer, norm in zip(self.layers, self.norms):
                x = x + layer(norm(x))
        # Drop the seed portion so the prediction-only shape is preserved
        # for the decoder head.

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
        """Parallel forward pass over latent (downsampled) frames.

        Args:
            inputs: token ids (B, S) or list of text strings.
            motion_length: target output motion length in raw frames. Internally
                           converted to latent frames via rvq_down_t stride.
            target_tokens: ground-truth (B, T', K) codebook indices for
                           teacher-forced training of the ResidualKHead arch.
                           Ignored by the legacy independent head.
            seed_latent: optional (B, P', d_model) pose-prefix conditioning;
                         see :meth:`forward_features` for semantics.

        Returns:
            logits: (B, T', K, V) -- per latent frame, per codebook, token distribution
            length_pred: (B,) predicted length in raw frames
        """
        x, cond = self.forward_features(inputs, motion_length, seed_latent=seed_latent)
        return self.decoder(x, cond, target_tokens=target_tokens)

    # ------------------------------------------------------------------ #
    # Streaming inference: one latent step at a time, constant memory.    #
    # See src/modules/motion/streaming.py for StreamingState docs.        #
    # Requires ``config.bidirectional=False`` -- BiMambaLayer has no      #
    # causal step path so streaming is unidirectional-only.               #
    # ------------------------------------------------------------------ #
    def stream_begin(
        self, inputs: torch.Tensor | list[str]
    ) -> StreamingState:
        """Initialise streaming state for a new action.

        Encodes the text once and allocates zero-initialised SSM hidden
        states for every Mamba layer. The returned state is mutated in
        place by :func:`stream_step`. To continue an existing action with a
        new instruction (action-to-action transition), call
        :meth:`StreamingState.carry_over` on the returned state instead of
        re-calling ``stream_begin``.
        """
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
        h0 = [
            torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
            for _ in self.layers
        ]

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
        """Emit ONE latent step.

        Args:
            state: persistent streaming state from :meth:`stream_begin`.
            target_tokens: per-step ``(B, 1, K)`` teacher-forced tokens for
                the AR K-head; ignored by the independent head. Streaming
                inference normally passes ``None`` and samples from the
                emitted logits separately.

        Returns:
            logits: ``(B, 1, K, V)`` codebook distribution for this step.
            length_pred: ``(B,)`` predicted length in raw frames (decoder
                emits this on every step; caller usually keeps only the
                last one).
            state: mutated state with ``latent_step`` incremented by 1 and
                the per-layer SSM state advanced.
        """
        t = state.latent_step

        if t >= state.max_steps:
            raise RuntimeError(
                f"stream_step() past max_steps={state.max_steps}; "
                "begin a new action via carry_over() or stream_begin()"
            )
        pos_id = self.motion_pos_ids[t : t + 1]  # type: ignore[index]
        pos = self.pos_embed(pos_id)  # (1, d_model)
        # initial latent for this step: cond + positional embedding
        x = state.cond + pos  # (B, d_model)

        for i, layer in enumerate(self.layers):
            if not isinstance(layer, MambaLayer):  # safety net
                raise RuntimeError(
                    "stream_step requires MambaLayer; got "
                    f"{type(layer).__name__} at layer {i}"
                )

            if self.use_film:
                x_norm = self.films[i](x.unsqueeze(1), state.cond).squeeze(1)
            else:
                x_norm = self.norms[i](x)
            out_t, h_new, conv_buf_new = layer.step(
                x_norm, state.layer_h[i], state.layer_conv[i]
            )
            state.layer_h[i] = h_new
            state.layer_conv[i] = conv_buf_new
            x = x + out_t  # residual connection mirrors the parallel forward

        # Decoder expects (B, T'=1, d_model); cond stays the same shape it
        # had in the parallel path. Independent head ignores target_tokens.
        logits, length_pred = self.decoder(
            x.unsqueeze(1), state.cond, target_tokens=target_tokens,
        )
        state.latent_step = t + 1

        return logits, length_pred, state
