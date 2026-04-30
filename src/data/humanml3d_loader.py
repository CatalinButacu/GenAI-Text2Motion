from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.amass import AMASSLoader
from src.data.augmentation import (
    AugmentationPipeline,
    canonicalizeRoot,
    mirrorFlip,
    mirrorFlipText,
    qualityFilter,
)
from src.data.motion_normalize import MotionStats, computeMotionStats, normalize
from src.shared.tokenizer import buildVocab
from src.shared.tokenizer import tokenize as tok

from .humanml3d import (
    DEFAULT_DIR,
    HumanML3DLoader,
    buildNormMap,
    loadMotionItem,
    normalizeAmassPath,
    parseIndexCsv,
    preloadSmplx,
)

__all__ = ["HumanML3DLoader", "normalizeAmassPath", "HumanML3DMotionDataset"]

log = logging.getLogger(__name__)


class HumanML3DMotionDataset(Dataset):
    def __init__(self, dataDir=DEFAULT_DIR, split="train", maxMotionLength=200,
                 maxTextLength=64, augment=False, vocab=None, preload=True,
                 stats: MotionStats | None = None, seed: int = 42,
                 mirrorProb: float = 0.5, amassDir: str = "data/AMASS",
                 minFrames: int = 30, applyQualityFilter: bool = True) -> None:
        self.dir = Path(dataDir)
        self.amassDir = Path(amassDir)
        self.max_motion, self.max_text = maxMotionLength, maxTextLength
        self.aug_pipeline = AugmentationPipeline(maxLength=maxMotionLength) if augment else None
        self.split = split
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        self.mirrorProb = mirrorProb if augment and split == "train" else 0.0
        self.minFrames = minFrames
        self.applyQualityFilter = applyQualityFilter
        self.samples = self.loadSamples(split)

        allTextsFlat = [t for s in self.samples for t in s["texts"]]
        self.vocab = vocab if vocab is not None else buildVocab(allTextsFlat)

        self.cache = self.preload(preload)
        self.motion_stats = stats if stats is not None else self.computeStats()

    def loadSamples(self, split: str) -> list:
        loader = HumanML3DLoader(str(self.dir))
        splitIds = set(loader.loadSplit(split))
        allTexts = {item["clip_id"]: item["texts"] for item in loader.loadTexts()}
        idxF = self.dir / "index.csv"

        if not idxF.exists():
            raise FileNotFoundError(f"{idxF} missing.")

        amass = AMASSLoader(str(self.amassDir))
        normMap = buildNormMap(amass.discoverFiles(), amass.dataDir)

        return parseIndexCsv(idxF, splitIds, allTexts, normMap)

    def preload(self, preload: bool) -> dict | None:
        if not preload:
            return None

        log.info("[HumanML3D] preloading %d samples...", len(self.samples))
        cache, bad = preloadSmplx(self.samples, self.amassDir)

        if bad:
            badS = set(bad)
            self.samples = [s for s in self.samples if s["clip_id"] not in badS]

        # Apply the same hygiene the unified loader does so train/eval see the
        # same effective dataset regardless of --data-source. minFrames drops
        # sub-second subclips; qualityFilter rejects static / kinematically
        # implausible motion. Both checks match unified_load.loadHumanml3d.
        keep: list[dict] = []
        nShort = nQuality = 0
        filteredCache: dict = {}

        for s in self.samples:
            clipId = s["clip_id"]
            m = cache.get(clipId)

            if m is None:
                continue

            if m.shape[0] < self.minFrames:
                nShort += 1
                continue

            if self.applyQualityFilter and not qualityFilter(m, 30.0):
                nQuality += 1
                continue
            filteredCache[clipId] = canonicalizeRoot(m)
            keep.append(s)
        self.samples = keep
        log.info(
            "[HumanML3D] %s: kept %d  (short<%d: %d, qualityRejected: %d, preloadBad: %d)",
            self.split, len(keep), self.minFrames, nShort, nQuality, len(bad or []),
        )

        return filteredCache

    def computeStats(self) -> MotionStats:
        """Compute normalisation stats over this split's loaded motion clips."""
        if self.cache is None or not self.cache:
            raise RuntimeError(
                "HumanML3DMotionDataset: cannot compute stats without preloaded cache"
            )
        samplesForStats = [{"motion": m} for m in self.cache.values()]
        s = computeMotionStats(samplesForStats)
        log.info("[HumanML3D] motion stats: mean|.|=%.4f std|.|=%.4f",
                 float(np.abs(s.mean).mean()), float(np.abs(s.std).mean()))

        return s

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        motion = loadMotionItem(s, self.cache, self.amassDir)
        text = (
            s["texts"][self.rng.integers(0, len(s["texts"]))]
            if self.split == "train" else s["text"]
        )

        if self.mirrorProb > 0.0 and self.rng.random() < self.mirrorProb:
            motion = mirrorFlip(motion)
            text = mirrorFlipText(text)

        if self.aug_pipeline is not None:
            motion = self.aug_pipeline(motion)
        motion = normalize(motion, self.motion_stats)
        T = min(motion.shape[0], self.max_motion)
        motion = np.pad(motion[:T], ((0, self.max_motion - T), (0, 0)))
        mask = np.zeros(self.max_motion, dtype=np.float32)
        mask[:T] = 1.0
        tokenIds = tok(text, self.vocab, maxLen=self.max_text)

        return {"token_ids": torch.tensor(tokenIds, dtype=torch.long),
                "texts": text,
                "motion": torch.tensor(motion, dtype=torch.float32),
                "motion_mask": torch.tensor(mask, dtype=torch.float32),
                "length": T}
