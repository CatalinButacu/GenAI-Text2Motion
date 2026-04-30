from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import AugmentationPipeline, mirrorFlip, mirrorFlipText
from src.data.motion_normalize import MotionStats, normalize
from src.shared.tokenizer import buildVocab, tokenize

from .dataset_cache import INGEST_MAX_LENGTH, loadOrBuildCache

log = logging.getLogger(__name__)


def encodeMotionSample(
    s: dict,
    augPipeline: AugmentationPipeline | None,
    maxMotionLength: int,
    maxTextLength: int,
    vocab: dict[str, int],
    stats: MotionStats | None = None,
    text: str | None = None,
) -> dict:
    """Encode a single motion sample dict into a model-ready tensor batch.

    If ``stats`` is provided, motion is z-score normalised per channel so the
    MSE loss is not dominated by high-variance root translation.
    If ``text`` is provided, it overrides ``s['text']`` -- used by datasets
    that sample uniformly from a per-clip annotation list on each epoch.
    """
    motion = s["motion"].copy()

    if augPipeline is not None:
        motion = augPipeline(motion)

    if stats is not None:
        motion = normalize(motion, stats)
    T = min(motion.shape[0], maxMotionLength)
    motion = np.pad(motion[:T], ((0, maxMotionLength - T), (0, 0)))
    mask = np.zeros(maxMotionLength, dtype=np.float32)
    mask[:T] = 1.0
    text = text if text is not None else s["text"]
    tokenIds = tokenize(text, vocab, maxLen=maxTextLength)

    return {
        "token_ids": torch.tensor(tokenIds, dtype=torch.long),
        "texts": text,
        "motion": torch.tensor(motion, dtype=torch.float32),
        "motion_mask": torch.tensor(mask, dtype=torch.float32),
        "length": T,
    }


class MotionDataset(Dataset):
    def __init__(
        self,
        dataDir: str = "data/AMASS",
        split: str = "train",
        maxMotionLength: int = 200,
        maxTextLength: int = 64,
        augment: bool = False,
        vocab: dict[str, int] | None = None,
        maxSamples: int | None = None,
        stats: MotionStats | None = None,
        mirrorProb: float = 0.5,
        seed: int = 42,
    ):
        self.maxMotionLength = maxMotionLength
        self.maxTextLength = maxTextLength
        self.aug_pipeline = None
        self.mirrorProb = mirrorProb if augment and split == "train" else 0.0
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1))

        if augment:
            self.aug_pipeline = AugmentationPipeline(maxLength=maxMotionLength)

        self.samples, cache_stats = loadOrBuildCache(dataDir, INGEST_MAX_LENGTH, maxSamples)
        self.motion_stats = stats if stats is not None else cache_stats

        rngSplit = np.random.default_rng(seed)
        rngSplit.shuffle(self.samples)  # type: ignore[arg-type]
        n = len(self.samples)
        tEnd = int(n * 0.8)   # 80 % train
        vEnd = int(n * 0.9)   # 10 % val, 10 % test
        if split == "train":
            self.samples = self.samples[:tEnd]
        elif split == "val":
            self.samples = self.samples[tEnd:vEnd]
        else:  # test -- held-out, never used for model selection
            self.samples = self.samples[vEnd:]

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = buildVocab([s["text"] for s in self.samples])

        log.info("[MotionDataset] %s: %d samples, vocab=%d, augment=%s, mirror_p=%.2f",
                 split, len(self.samples), len(self.vocab), augment, self.mirrorProb)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.mirrorProb > 0.0 and self.rng.random() < self.mirrorProb:
            s = {**s, "motion": mirrorFlip(s["motion"]), "text": mirrorFlipText(s["text"])}

        return encodeMotionSample(
            s, self.aug_pipeline,
            self.maxMotionLength, self.maxTextLength, self.vocab,
            stats=self.motion_stats,
        )
