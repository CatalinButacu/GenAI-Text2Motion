from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import AugmentationPipeline, detectTpose, qualityFilter, resampleToFps
from src.data.motion_normalize import MotionStats, computeMotionStats, normalize
from src.shared.tokenizer import buildVocab

from .motion_dataset import encodeMotionSample
from .unified import buildSourcesBuffer, splitSamples

log = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    enabled: bool = True
    maxSamples: int | None = None
    dataDir: str = ""       # path to dataset root; required by source loaders
    amassDir: str = "data/AMASS"  # AMASS backing store (HumanML3D needs this)


@dataclass
class UnifiedConfig:
    amass: SourceConfig = field(
        default_factory=lambda: SourceConfig(dataDir="data/AMASS")
    )
    arctic: SourceConfig = field(
        default_factory=lambda: SourceConfig(dataDir="data/ARCTIC/unpack")
    )
    # HumanML3D off by default -- requires index.csv + AMASS backing
    humanml3d: SourceConfig = field(
        default_factory=lambda: SourceConfig(enabled=False, dataDir="data/humanml3d")
    )


class UnifiedMotionDataset(Dataset):
    def __init__(self, split="train", maxMotionLength=200, maxTextLength=64,
                 augment=False, vocab=None, stats: MotionStats | None = None,
                 config=None, minFrames=30, preloadedBuf=None):
        self.maxMotionLength = maxMotionLength
        self.maxTextLength = maxTextLength
        self.aug_pipeline = None

        if augment:
            self.aug_pipeline = AugmentationPipeline(maxLength=maxMotionLength)

        if preloadedBuf is not None:
            buf = preloadedBuf
        else:
            cfg = config or UnifiedConfig()
            buf = buildSourcesBuffer(cfg, minFrames, resampleToFps, qualityFilter, detectTpose)
        counts: dict[str, int] = {}

        for s in buf:
            counts[s.get("source", "?")] = counts.get(s.get("source", "?"), 0) + 1
        log.info("[UnifiedDataset] total: %d  %s", len(buf), counts)
        self.samples = splitSamples(buf, split)

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = buildVocab([s["text"] for s in self.samples])

        # Compute normalization stats from this split so motion is z-score normalised
        if stats is not None:
            self.motion_stats: MotionStats = stats
        else:
            self.motion_stats = computeMotionStats(self.samples)

        log.info("[UnifiedDataset] %s: %d samples, vocab=%d",
                 split, len(self.samples), len(self.vocab))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = encodeMotionSample(
            s, self.aug_pipeline, self.maxMotionLength, self.maxTextLength,
            self.vocab, stats=self.motion_stats,
        )
        item["source"] = s.get("source", "unknown")

        return item

    @property
    def sourceCounts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for s in self.samples:
            src = s.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1

        return counts

