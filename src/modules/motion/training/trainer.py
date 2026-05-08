from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

import joblib
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.augmentation import detectTpose, qualityFilter, resampleToFps
from src.data.humanml3d_loader import HumanML3DMotionDataset
from src.data.motion_dataset import MotionDataset
from src.data.unified import buildSourcesBuffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig, UnifiedMotionDataset

from ..config import TrainingConfig
from .base_trainer import BaseSSMTrainer

log = logging.getLogger(__name__)

# dataset_factory(config, max_samples) -> (train_ds, val_ds, test_ds, train_sampler_or_None)
DatasetFactory = Callable[[TrainingConfig, int | None], tuple]


def buildUnifiedBuf(cfg, minFrames: int = 30) -> list[dict]:
    """Load and preprocess all data sources into a shared buffer, with joblib cache."""
    # Cache key includes which sources are enabled and their data dirs
    sourcesKey = "|".join(
        f"{name}:{getattr(getattr(cfg, name, None), 'enabled', False)}"
        f":{getattr(getattr(cfg, name, None), 'dataDir', '')}"
        f":{getattr(getattr(cfg, name, None), 'maxSamples', None)}"
        for name in ("amass", "arctic", "humanml3d", "interx")
    )
    ck = hashlib.md5(f"{sourcesKey}:{minFrames}".encode()).hexdigest()[:12]
    cachePath = Path("data/.cache") / f"unified_buf_{ck}.joblib"

    if cachePath.exists():
        log.info("[unified] loading buffer from cache: %s", cachePath)
        buf = joblib.load(cachePath)
        log.info("[unified] cache hit: %d samples", len(buf))

        return buf

    log.info("[unified] building buffer (first time, will cache to %s) ...", cachePath)
    buf = buildSourcesBuffer(cfg, minFrames, resampleToFps, qualityFilter, detectTpose)
    os.makedirs(cachePath.parent, exist_ok=True)
    joblib.dump(buf, cachePath, compress=3)
    log.info("[unified] cached %d processed samples -> %s", len(buf), cachePath)

    return buf


class SSMTrainer(BaseSSMTrainer):
    """Generic SSM trainer parameterised by a dataset factory.

    The factory returns ``(train_ds, val_ds, sampler)`` where ``sampler`` is
    optional (used by the unified multi-source pipeline for source balancing).
    """

    def __init__(
        self,
        config: TrainingConfig,
        datasetFactory: DatasetFactory,
        dataSource: str = "unknown",
    ) -> None:
        self.config = config
        self.dataSource = dataSource

        if config.seed is not None:
            self.lockSeed(config.seed)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.trainDs, self.valDs, self.testDs, trainSampler = datasetFactory(
            config, config.maxSamples
        )
        config.vocabSize = len(self.trainDs.vocab)
        log.info(
            "[%s] train=%d val=%d test=%d vocab=%d",
            dataSource,
            len(self.trainDs),
            len(self.valDs),
            len(self.testDs),
            config.vocabSize,
        )
        self.finalizeInit(config, trainSampler=trainSampler)

    def saveCheckpoint(self, epoch: int, valLoss: float, isBest: bool) -> None:
        ssmKeys = {
            k: v
            for k, v in self.model.state_dict().items()
            if k.startswith("layers.")
            or k.startswith("films.")
            or k.startswith("decoder.")
            or k.startswith("pos_embed.")
            or k.startswith("condition_proj.")
        }
        ck = {
            "epoch": epoch,
            "global_step": self.step,
            "model_state_dict": self.model.state_dict(),
            "motion_ssm_state_dict": ssmKeys,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "valLoss": valLoss,
            "config": self.config,
            "vocab": self.trainDs.vocab,
            "motion_stats": self.trainDs.motion_stats,
            "data_source": self.dataSource,
        }
        path = os.path.join(self.config.checkpointDir, f"checkpoint_epoch{epoch}.pt")
        self.saveCheckpointAt(path, ck)

        if isBest:
            self.saveCheckpointAt(os.path.join(self.config.checkpointDir, "best_model.pt"), ck)
            log.info("[%s] best model saved: valLoss=%.4f", self.dataSource, valLoss)


# ---- Dataset factories ------------------------------------------------------


def amassFactory(config: TrainingConfig, maxSamples):
    train = MotionDataset(
        config.dataDir, "train", config.maxMotionLength,
        augment=True, maxSamples=maxSamples,
    )
    val = MotionDataset(
        config.dataDir, "val", config.maxMotionLength,
        vocab=train.vocab, maxSamples=maxSamples, stats=train.motion_stats,
    )
    test = MotionDataset(
        config.dataDir, "test", config.maxMotionLength,
        vocab=train.vocab, maxSamples=maxSamples, stats=train.motion_stats,
    )
    return train, val, test, None


def humanml3dFactory(config: TrainingConfig, maxSamples):
    amassDir = getattr(config, "amass_dir", "data/AMASS")
    train = HumanML3DMotionDataset(
        dataDir=config.dataDir, amassDir=amassDir,
        split="train", maxMotionLength=config.maxMotionLength,
        maxTextLength=config.maxTextLength, augment=True,
    )
    val = HumanML3DMotionDataset(
        dataDir=config.dataDir, amassDir=amassDir,
        split="val", maxMotionLength=config.maxMotionLength,
        maxTextLength=config.maxTextLength,
        vocab=train.vocab, stats=train.motion_stats,
    )
    test = HumanML3DMotionDataset(
        dataDir=config.dataDir, amassDir=amassDir,
        split="test", maxMotionLength=config.maxMotionLength,
        maxTextLength=config.maxTextLength,
        vocab=train.vocab, stats=train.motion_stats,
    )
    return train, val, test, None


def unifiedFactory(config: TrainingConfig, maxSamples):
    sources = getattr(config, "unifiedSources", ["amass", "arctic"])

    def makeSrc(enabled: bool, dataDir: str, amassDir: str = "data/AMASS") -> SourceConfig:
        sc = SourceConfig(enabled=enabled, dataDir=dataDir, amassDir=amassDir)
        if maxSamples is not None:
            sc.maxSamples = maxSamples
        return sc

    cfg = UnifiedConfig(
        amass=makeSrc("amass" in sources, getattr(config, "dataDir", "data/AMASS")),
        arctic=makeSrc(
            "arctic" in sources,
            getattr(config, "arcticDataDir", "data/arctic/unpack"),
        ),
        humanml3d=makeSrc(
            "humanml3d" in sources,
            getattr(config, "humanml3dDir", "data/humanml3d"),
            amassDir=getattr(config, "amassDir", "data/AMASS"),
        ),
        interx=makeSrc(
            "interx" in sources,
            getattr(config, "interxDir", "data/inter-x"),
        ),
    )
    sharedBuf = buildUnifiedBuf(cfg)
    log.info("[unified] shared buffer: %d samples  sources=%s", len(sharedBuf), sources)

    train = UnifiedMotionDataset(
        split="train",
        maxMotionLength=config.maxMotionLength,
        maxTextLength=config.maxTextLength,
        augment=True,
        preloadedBuf=sharedBuf,
    )
    val = UnifiedMotionDataset(
        split="val",
        maxMotionLength=config.maxMotionLength,
        maxTextLength=config.maxTextLength,
        vocab=train.vocab,
        stats=train.motion_stats,
        preloadedBuf=sharedBuf,
    )
    test = UnifiedMotionDataset(
        split="test",
        maxMotionLength=config.maxMotionLength,
        maxTextLength=config.maxTextLength,
        vocab=train.vocab,
        stats=train.motion_stats,
        preloadedBuf=sharedBuf,
    )
    counts = train.sourceCounts
    srcWt = {src: 1.0 / max(n, 1) for src, n in counts.items()}
    weights = [srcWt.get(s.get("source", "unknown"), 1.0) for s in train.samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return train, val, test, sampler


# ---- Runner -----------------------------------------------------------------


def run(config: TrainingConfig | None, factory, label: str) -> float:
    cfg = config or TrainingConfig()
    log.info("[%s] using SSMTrainer (RVQ token cross-entropy)", label)

    return SSMTrainer(cfg, factory, label).train()


# ---- Entry points -----------------------------------------------------------


def trainAmass(config: TrainingConfig | None = None) -> float:
    return run(config, amassFactory, "amass")


def trainHumanml3d(config: TrainingConfig | None = None) -> float:
    return run(config, humanml3dFactory, "humanml3d")


def trainUnified(config: TrainingConfig | None = None) -> float:
    return run(config, unifiedFactory, "unified")
