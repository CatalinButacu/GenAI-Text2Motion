from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import (
    AugmentationPipeline,
    detect_tpose,
    quality_filter,
    resample_to_fps,
)
from src.data.motion_normalize import MotionStats, compute_motion_stats, normalize
from src.shared.tokenizer import build_vocab

from .motion_dataset import encode_motion_sample
from .unified import build_sources_buffer, split_samples

log = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    enabled: bool = True
    max_samples: int | None = None
    data_dir: str = ""       # path to dataset root; required by source loaders
    amass_dir: str = "data/AMASS"  # AMASS backing store (HumanML3D needs this)


@dataclass
class UnifiedConfig:
    amass: SourceConfig = field(
        default_factory=lambda: SourceConfig(data_dir="data/AMASS")
    )
    arctic: SourceConfig = field(
        default_factory=lambda: SourceConfig(enabled=False, data_dir="data/arctic/unpack")
    )
    # HumanML3D off by default -- requires index.csv + AMASS backing
    humanml3d: SourceConfig = field(
        default_factory=lambda: SourceConfig(enabled=False, data_dir="data/humanml3d")
    )
    interx: SourceConfig = field(
        default_factory=lambda: SourceConfig(enabled=False, data_dir="data/inter-x")
    )


class UnifiedMotionDataset(Dataset):
    def __init__(self, split="train", max_motion_length=200, max_text_length=64,
                 augment=False, vocab=None, stats: MotionStats | None = None,
                 config=None, min_frames=30, preloaded_buf=None,
                 trans_stats_by_source: dict[str, MotionStats] | None = None):
        self.max_motion_length = max_motion_length
        self.max_text_length = max_text_length
        self.aug_pipeline = None
        self.trans_stats_by_source = trans_stats_by_source

        if augment:
            self.aug_pipeline = AugmentationPipeline(max_length=max_motion_length)

        if preloaded_buf is not None:
            buf = preloaded_buf
        else:
            cfg = config or UnifiedConfig()
            buf = build_sources_buffer(
                cfg, min_frames, resample_to_fps, quality_filter, detect_tpose,
            )
        counts: dict[str, int] = {}

        for s in buf:
            counts[s.get("source", "?")] = counts.get(s.get("source", "?"), 0) + 1
        log.info("[UnifiedDataset] total: %d  %s", len(buf), counts)
        self.samples = split_samples(buf, split)

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = build_vocab([s["text"] for s in self.samples])

        # Compute normalization stats from this split so motion is z-score normalised.
        # WARNING: for val/test you should pass stats= from the train split so all
        # splits see the same normalization. Computing per-split stats produces
        # silent train/eval distribution drift -- the recurring bug from
        # project_cloud_lessons_2026_05_13.md.
        if stats is not None:
            self.motion_stats: MotionStats = stats
        else:
            if split != "train":
                log.warning(
                    "[UnifiedDataset] split=%r built with no stats= argument; "
                    "computing per-split stats. Pass stats= from the train split "
                    "to avoid train/eval normalization drift.", split,
                )
            self.motion_stats = compute_motion_stats(self.samples)

        log.info("[UnifiedDataset] %s: %d samples, vocab=%d",
                 split, len(self.samples), len(self.vocab))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = encode_motion_sample(
            s, self.aug_pipeline, self.max_motion_length, self.max_text_length,
            self.vocab, stats=self.motion_stats,
            trans_stats_by_source=self.trans_stats_by_source,
        )
        item["source"] = s.get("source", "unknown")

        return item

    @property
    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for s in self.samples:
            src = s.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1

        return counts

