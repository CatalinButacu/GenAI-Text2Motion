from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

import joblib
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.augmentation import detect_tpose, quality_filter, resample_to_fps
from src.data.humanml3d_loader import HumanML3DMotionDataset
from src.data.motion_dataset import MotionDataset
from src.data.unified import build_sources_buffer
from src.data.unified_dataset import SourceConfig, UnifiedConfig, UnifiedMotionDataset
from src.shared.config import TrainingConfig
from src.shared.constants import TRAINING

from .base_trainer import BaseSSMTrainer

log = logging.getLogger(__name__)

# dataset_factory(config, max_samples) -> (train_ds, val_ds, test_ds, train_sampler_or_None)
DatasetFactory = Callable[[TrainingConfig, int | None], tuple]


def build_unified_buf(cfg, min_frames: int = TRAINING.min_quality_frames) -> list[dict]:
    """Load and preprocess all data sources into a shared buffer, with joblib cache."""

    def key_for(name: str) -> str:
        sc = getattr(cfg, name, None)

        return (
            f"{name}:{getattr(sc, 'enabled', False)}"
            f":{getattr(sc, 'data_dir', '')}"
            f":{getattr(sc, 'max_samples', None)}"
        )

    sources_key = "|".join(key_for(n) for n in ("amass", "arctic", "humanml3d", "interx"))
    ck = hashlib.md5(f"{sources_key}:{min_frames}".encode()).hexdigest()[: TRAINING.cache_hash_len]
    cache_path = Path("data/.cache") / f"unified_buf_{ck}.joblib"

    if cache_path.exists():
        log.info("loading buffer from cache: %s", cache_path)
        buf = joblib.load(cache_path)
        log.info("cache hit: %d samples", len(buf))

        return buf

    log.info("building buffer (first time, will cache to %s) ...", cache_path)
    buf = build_sources_buffer(cfg, min_frames, resample_to_fps, quality_filter, detect_tpose)
    os.makedirs(cache_path.parent, exist_ok=True)
    joblib.dump(buf, cache_path, compress=TRAINING.joblib_compress_level)
    log.info("cached %d processed samples -> %s", len(buf), cache_path)

    return buf


class SSMTrainer(BaseSSMTrainer):
    """Generic SSM trainer parameterised by a dataset factory.
    Factory returns (train_ds, val_ds, test_ds, sampler) -- sampler optional."""

    def __init__(
        self,
        config: TrainingConfig,
        dataset_factory: DatasetFactory,
        data_source: str = "unknown",
    ) -> None:
        self.config = config
        self.data_source = data_source

        if config.seed is not None:
            self.lock_seed(config.seed)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.train_ds, self.val_ds, self.test_ds, train_sampler = dataset_factory(
            config, config.max_samples
        )
        config.vocab_size = len(self.train_ds.vocab)
        log.info(
            "[%s] train=%d val=%d test=%d vocab=%d",
            data_source,
            len(self.train_ds),
            len(self.val_ds),
            len(self.test_ds),
            config.vocab_size,
        )
        self.finalize_init(config, train_sampler=train_sampler)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        ssm_keys = {
            k: v
            for k, v in self.model.state_dict().items()
            if k.startswith(("layers.", "films.", "decoder.", "pos_embed.", "condition_proj."))
        }
        ck = {
            "epoch": epoch,
            "global_step": self.step,
            "model_state_dict": self.model.state_dict(),
            "motion_ssm_state_dict": ssm_keys,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "vocab": self.train_ds.vocab,
            "motion_stats": self.train_ds.motion_stats,
            "data_source": self.data_source,
            "streaming_capable": not self.model.bidirectional,
        }
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
        self.save_checkpoint_at(path, ck)

        if is_best:
            self.save_checkpoint_at(os.path.join(self.config.checkpoint_dir, "best_model.pt"), ck)
            log.info("[%s] best model saved: val_loss=%.4f", self.data_source, val_loss)


def amass_factory(config: TrainingConfig, max_samples):
    train = MotionDataset(
        config.data_dir,
        "train",
        config.max_motion_length,
        augment=True,
        max_samples=max_samples,
    )
    val = MotionDataset(
        config.data_dir,
        "val",
        config.max_motion_length,
        vocab=train.vocab,
        max_samples=max_samples,
        stats=train.motion_stats,
    )
    test = MotionDataset(
        config.data_dir,
        "test",
        config.max_motion_length,
        vocab=train.vocab,
        max_samples=max_samples,
        stats=train.motion_stats,
    )
    return train, val, test, None


def humanml3d_factory(config: TrainingConfig, max_samples):
    amass_dir = getattr(config, "amass_dir", TRAINING.default_amass_dir)
    train = HumanML3DMotionDataset(
        data_dir=config.data_dir,
        amass_dir=amass_dir,
        split="train",
        max_motion_length=config.max_motion_length,
        max_text_length=config.max_text_length,
        augment=True,
    )
    val = HumanML3DMotionDataset(
        data_dir=config.data_dir,
        amass_dir=amass_dir,
        split="val",
        max_motion_length=config.max_motion_length,
        max_text_length=config.max_text_length,
        vocab=train.vocab,
        stats=train.motion_stats,
    )
    test = HumanML3DMotionDataset(
        data_dir=config.data_dir,
        amass_dir=amass_dir,
        split="test",
        max_motion_length=config.max_motion_length,
        max_text_length=config.max_text_length,
        vocab=train.vocab,
        stats=train.motion_stats,
    )
    return train, val, test, None


def unified_factory(config: TrainingConfig, max_samples):
    sources = getattr(config, "unified_sources", list(TRAINING.default_unified_sources))

    def make_src(
        enabled: bool, data_dir: str, amass_dir: str = TRAINING.default_amass_dir
    ) -> SourceConfig:
        sc = SourceConfig(enabled=enabled, data_dir=data_dir, amass_dir=amass_dir)
        if max_samples is not None:
            sc.max_samples = max_samples
        return sc

    cfg = UnifiedConfig(
        amass=make_src("amass" in sources, getattr(config, "data_dir", TRAINING.default_amass_dir)),
        arctic=make_src(
            "arctic" in sources,
            getattr(config, "arctic_data_dir", TRAINING.default_arctic_dir),
        ),
        humanml3d=make_src(
            "humanml3d" in sources,
            getattr(config, "humanml3d_dir", TRAINING.default_humanml3d_dir),
            amass_dir=getattr(config, "amass_dir", TRAINING.default_amass_dir),
        ),
        interx=make_src(
            "interx" in sources,
            getattr(config, "interx_dir", TRAINING.default_interx_dir),
        ),
    )
    shared_buf = build_unified_buf(cfg)
    log.info("shared buffer: %d samples  sources=%s", len(shared_buf), sources)

    train = UnifiedMotionDataset(
        split="train",
        max_motion_length=config.max_motion_length,
        max_text_length=config.max_text_length,
        augment=True,
        preloaded_buf=shared_buf,
    )
    val = UnifiedMotionDataset(
        split="val",
        max_motion_length=config.max_motion_length,
        max_text_length=config.max_text_length,
        vocab=train.vocab,
        stats=train.motion_stats,
        preloaded_buf=shared_buf,
    )
    test = UnifiedMotionDataset(
        split="test",
        max_motion_length=config.max_motion_length,
        max_text_length=config.max_text_length,
        vocab=train.vocab,
        stats=train.motion_stats,
        preloaded_buf=shared_buf,
    )
    counts = train.source_counts
    src_wt = {src: 1.0 / max(n, 1) for src, n in counts.items()}
    weights = [src_wt.get(s.get("source", "unknown"), 1.0) for s in train.samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return train, val, test, sampler


def run(config: TrainingConfig | None, factory, label: str) -> float:
    cfg = config or TrainingConfig()
    log.info("[%s] using SSMTrainer (RVQ token cross-entropy)", label)

    return SSMTrainer(cfg, factory, label).train()


def train_amass(config: TrainingConfig | None = None) -> float:
    return run(config, amass_factory, "amass")


def train_humanml3d(config: TrainingConfig | None = None) -> float:
    return run(config, humanml3d_factory, "humanml3d")


def train_unified(config: TrainingConfig | None = None) -> float:
    return run(config, unified_factory, "unified")
