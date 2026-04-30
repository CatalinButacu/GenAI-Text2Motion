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
from src.utils.mem_profile import profileMemory

from .models import MotionClip, MotionSource

log = logging.getLogger(__name__)


def sampleIndices(
    logits: torch.Tensor,
    temperature: float = 1.0,
    topP: float = 1.0,
) -> torch.Tensor:
    """Sample token indices from (B, T', K, V) logits.

    temperature=1.0 and topP=1.0 is identical to argmax (greedy/deterministic).
    temperature>1 diversifies output; topP<1 applies nucleus (top-p) filtering.
    """
    if temperature <= 0.0 or (temperature == 1.0 and topP >= 1.0):
        return logits.argmax(dim=-1)

    B, T, K, V = logits.shape
    scaled = logits / temperature

    if topP < 1.0:
        probs = F.softmax(scaled, dim=-1)
        sortedProbs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumprobs = sortedProbs.cumsum(dim=-1)
        # Zero out tokens beyond the top-p nucleus (keep at least 1 token per position)
        remove = (cumprobs - sortedProbs) >= topP
        sortedProbs = sortedProbs.masked_fill(remove, 0.0)
        probs.scatter_(-1, sorted_idx, sortedProbs)
        flat = probs.reshape(B * T * K, V)
    else:
        flat = F.softmax(scaled.reshape(B * T * K, V), dim=-1)

    indices = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return indices.reshape(B, T, K)


def coerceConfig(raw) -> TrainingConfig:
    if isinstance(raw, TrainingConfig):
        return raw
    fieldVals = {f.name: getattr(raw, f.name) for f in dc_fields(raw)}
    return TrainingConfig(**fieldVals)


class SSMMotionModel:
    def __init__(
        self,
        checkpointPath: str = "checkpoints/motion_ssm/best_model.pt",
        rvqCheckpointPath: str = "checkpoints/rvq_tokenizer/best_model.pt",
        dataDir: str = "data/AMASS",
    ) -> None:
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(
                f"[SSM] checkpoint not found: {checkpointPath!r}. "
                "Train one via `python scripts/training/train_motion_ssm.py`."
            )
        if not os.path.exists(rvqCheckpointPath):
            raise FileNotFoundError(
                f"[SSM] RVQ tokenizer not found: {rvqCheckpointPath!r}. "
                "Train one via `python scripts/training/train_rvq_tokenizer.py`."
            )

        self.checkpointPath = checkpointPath
        self.rvqCheckpointPath = rvqCheckpointPath
        self.dataDir = dataDir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ck = torch.load(checkpointPath, map_location=self.device, weights_only=False)
        self.vocab: dict = ck["vocab"]
        cfg = coerceConfig(ck["config"])
        self.model = TextToMotionSSM(cfg).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

        rvqCk = torch.load(rvqCheckpointPath, map_location=self.device, weights_only=False)
        self.tokenizer = MotionRVQTokenizer(
            motionDim=cfg.motionDim,
            latentDim=cfg.rvqLatentDim,
            nCodebooks=cfg.rvqNCodebooks,
            codebookSize=cfg.rvqCodebookSize,
            downT=cfg.rvqDownT,
        ).to(self.device)
        self.tokenizer.load_state_dict(rvqCk["model_state_dict"])
        self.tokenizer.eval()

        log.info(
            "[SSM] loaded %s (valLoss=%s) + rvq %s (val_loss=%s)",
            checkpointPath,
            ck.get("valLoss", ck.get("val_loss", "N/A")),
            rvqCheckpointPath,
            rvqCk.get("val_loss", "N/A"),
        )

    @profileMemory
    def generateFromTextTokens(
        self,
        text: str,
        numFrames: int = 100,
        temperature: float = 1.0,
        topP: float = 1.0,
    ) -> MotionClip:
        if getattr(self.model.config, "useSbert", False):
            inputs = [text]
        else:
            inputs = (
                torch.tensor(tokenize(text, self.vocab), dtype=torch.long)
                .unsqueeze(0)
                .to(self.device)
            )

        with torch.no_grad():
            logits, _ = self.model(inputs, numFrames)  # (B, T', K, V)
            indices = sampleIndices(logits, temperature, topP)  # (B, T', K)
            motion = self.tokenizer.decode(indices)  # (B, T, motion_dim)
            motion = motion[:, :numFrames]  # trim to requested length
            motion = motion.cpu().numpy()[0]

        return MotionClip(action=text, smplxParams=motion, source=MotionSource.SSM)

    def invoke(self, text: str, durationS: float = 3.0) -> MotionClip:
        return self.generateFromTextTokens(text, int(durationS * 30))
