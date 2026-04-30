from __future__ import annotations

import logging

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.modules.motion.ssm import BiMambaLayer, MambaLayer, SSMConfig

log = logging.getLogger(__name__)


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocabSize: int, embedDim: int, maxLength: int):
        super().__init__()
        self.word_embed = nn.Embedding(vocabSize, embedDim, padding_idx=0)
        self.pos_embed = nn.Embedding(maxLength, embedDim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedDim, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.register_buffer("pos_ids", torch.arange(maxLength).unsqueeze(0))

    def forward(self, tokenIds: torch.Tensor) -> torch.Tensor:
        b, s = tokenIds.shape
        pos = self.pos_ids[:, :s].expand(b, -1)  # type: ignore[index]
        x = self.word_embed(tokenIds) + self.pos_embed(pos)

        return self.encoder(x).mean(dim=1)


class SBERTTextEncoder(nn.Module):
    SBERT_DIM = 384  # fixed by all-MiniLM-L6-v2

    def __init__(self, dModel: int, modelName: str = "all-MiniLM-L6-v2", freeze: bool = True):
        super().__init__()
        self.sbert = SentenceTransformer(modelName)

        if freeze:
            for param in self.sbert.parameters():
                param.requires_grad = False
        self.available = True
        log.info("SBERTTextEncoder: loaded %s (frozen=%s)", modelName, freeze)
        self.proj = nn.Sequential(nn.Linear(self.SBERT_DIM, dModel), nn.LayerNorm(dModel))

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


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: text-conditioned scale+shift at every SSM layer.

    Replaces plain LayerNorm with norm(x)*gamma + beta where gamma,beta are
    predicted from the text condition vector. Initialised so gamma=1, beta=0
    (identity at start of training).
    """

    def __init__(self, dModel: int, dCond: int):
        super().__init__()
        self.norm = nn.LayerNorm(dModel)
        self.proj = nn.Linear(dCond, dModel * 2)
        # Identity init: gamma->1, beta->0
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:dModel])
        nn.init.zeros_(self.proj.bias[dModel:])

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
        dModel: int,
        nCodebooks: int,
        codebookSize: int,
        maxLength: int,
    ):
        super().__init__()
        self.nCodebooks = nCodebooks
        self.codebookSize = codebookSize
        self.maxLength = maxLength
        self.length_head = nn.Linear(dModel, 1)
        self.token_heads = nn.ModuleList(
            [nn.Linear(dModel, codebookSize) for _ in range(nCodebooks)]
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> tuple:
        # features: (B, T', d_model), condition: (B, d_model)
        # returns: logits (B, T', K, V), length_pred (B,)
        logits = torch.stack([head(features) for head in self.token_heads], dim=2)
        lengthPred = torch.sigmoid(self.length_head(condition)).squeeze(-1) * self.maxLength

        return logits, lengthPred


class TextToMotionSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.useSbert:
            self.text_encoder = SBERTTextEncoder(
                dModel=config.dModel,
                modelName=config.sbertModel,
                freeze=config.freezeSbert,
            )
            self.condition_proj = nn.Identity()
        else:
            self.text_encoder = SimpleTextEncoder(
                config.vocabSize,
                config.textEmbedDim,
                config.maxTextLength,
            )
            self.condition_proj = nn.Linear(config.textEmbedDim, config.dModel)

        self.latent_length = config.maxMotionLength // config.rvqDownT
        self.pos_embed = nn.Embedding(self.latent_length, config.dModel)
        self.register_buffer("motion_pos_ids", torch.arange(self.latent_length))

        bidirectional = config.bidirectional
        gradCkpt = config.gradientCheckpointing
        useFilm = config.useFilm
        layerCls = BiMambaLayer if bidirectional else MambaLayer
        ssmCfg = SSMConfig(
            dModel=config.dModel, dState=config.dState, gradientCheckpointing=gradCkpt
        )
        self.layers = nn.ModuleList([layerCls(ssmCfg) for _ in range(config.nLayers)])
        self.useFilm = useFilm
        self.bidirectional = bidirectional
        self.ssm_cfg = ssmCfg

        if useFilm:
            self.films = nn.ModuleList(
                [FiLM(config.dModel, config.dModel) for _ in range(config.nLayers)]
            )
        else:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(config.dModel) for _ in range(config.nLayers)]
            )
        self.decoder = RVQMotionDecoder(
            dModel=config.dModel,
            nCodebooks=config.rvqNCodebooks,
            codebookSize=config.rvqCodebookSize,
            maxLength=config.maxMotionLength,
        )

    def forward(
        self,
        inputs: torch.Tensor | list[str],
        motionLength: int | None = None,
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

        if motionLength is None or motionLength > self.config.maxMotionLength:
            motionLength = self.config.maxMotionLength
        assert isinstance(motionLength, int)
        latentLen = max(1, motionLength // self.config.rvqDownT)
        pos = self.pos_embed(self.motion_pos_ids[:latentLen])  # type: ignore[index]
        x = cond.unsqueeze(1) + pos.unsqueeze(0)  # (B, T', d_model)

        if self.useFilm:
            for layer, film in zip(self.layers, self.films):
                x = x + layer(film(x, cond))
        else:
            for layer, norm in zip(self.layers, self.norms):
                x = x + layer(norm(x))

        return self.decoder(x, cond)
