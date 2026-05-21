from __future__ import annotations

import logging

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.modules.motion.ssm import BiMambaLayer, MambaLayer, SSMConfig

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

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> tuple:
        # features: (B, T', d_model), condition: (B, d_model)
        # returns: logits (B, T', K, V), length_pred (B,)
        logits = torch.stack([head(features) for head in self.token_heads], dim=2)
        length_pred = torch.sigmoid(self.length_head(condition)).squeeze(-1) * self.max_length

        return logits, length_pred


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
        self.decoder = RVQMotionDecoder(
            d_model=config.d_model,
            n_codebooks=config.rvq_n_codebooks,
            codebook_size=config.rvq_codebook_size,
            max_length=config.max_motion_length,
        )

    def forward(
        self,
        inputs: torch.Tensor | list[str],
        motion_length: int | None = None,
    ) -> tuple:
        """Parallel forward pass over latent (downsampled) frames.

        Args:
            inputs: token ids (B, S) or list of text strings.
            motion_length: target output motion length in raw frames. Internally
                           converted to latent frames via rvq_down_t stride.

        Returns:
            logits: (B, T', K, V) -- per latent frame, per codebook, token distribution
            length_pred: (B,) predicted length in raw frames
        """
        cond = self.condition_proj(self.text_encoder(inputs))

        if motion_length is None or motion_length > self.config.max_motion_length:
            motion_length = self.config.max_motion_length
        assert isinstance(motion_length, int)
        latent_len = max(1, motion_length // self.config.rvq_down_t)
        pos = self.pos_embed(self.motion_pos_ids[:latent_len])  # type: ignore[index]
        x = cond.unsqueeze(1) + pos.unsqueeze(0)  # (B, T', d_model)

        if self.use_film:
            for layer, film in zip(self.layers, self.films):
                x = x + layer(film(x, cond))
        else:
            for layer, norm in zip(self.layers, self.norms):
                x = x + layer(norm(x))

        return self.decoder(x, cond)
