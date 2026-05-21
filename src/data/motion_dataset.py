from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import AugmentationPipeline, mirror_flip, mirror_flip_text
from src.data.motion_normalize import MotionStats, normalize
from src.shared.tokenizer import build_vocab, tokenize

from .dataset_cache import INGEST_MAX_LENGTH, load_or_build_cache

log = logging.getLogger(__name__)


def encode_motion_sample(
    s: dict,
    aug_pipeline: AugmentationPipeline | None,
    max_motion_length: int,
    max_text_length: int,
    vocab: dict[str, int],
    stats: MotionStats | None = None,
    text: str | None = None,
    trans_stats_by_source: dict[str, MotionStats] | None = None,
) -> dict:
    """Encode a single motion sample dict into a model-ready tensor batch.

    If ``stats`` is provided, motion is z-score normalised per channel so the
    MSE loss is not dominated by high-variance root translation.
    If ``trans_stats_by_source`` is provided, channels 3:6 are normalized using
    the source-specific (mean, std) instead of the shared stats — fixes the
    bimodal AMASS-vs-HumanML3D translation distribution.
    If ``text`` is provided, it overrides ``s['text']`` -- used by datasets
    that sample uniformly from a per-clip annotation list on each epoch.
    """
    motion = s["motion"].copy()

    if aug_pipeline is not None:
        motion = aug_pipeline(motion)

    if stats is not None:
        src_key = s.get("source", "amass")
        trans_stats = trans_stats_by_source.get(src_key) if trans_stats_by_source else None
        motion = normalize(motion, stats, trans_stats=trans_stats)
    T = min(motion.shape[0], max_motion_length)
    motion = np.pad(motion[:T], ((0, max_motion_length - T), (0, 0)))
    mask = np.zeros(max_motion_length, dtype=np.float32)
    mask[:T] = 1.0
    text = text if text is not None else s["text"]
    token_ids = tokenize(text, vocab, max_len=max_text_length)

    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "texts": text,
        "motion": torch.tensor(motion, dtype=torch.float32),
        "motion_mask": torch.tensor(mask, dtype=torch.float32),
        "length": T,
    }


class MotionDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/AMASS",
        split: str = "train",
        max_motion_length: int = 200,
        max_text_length: int = 64,
        augment: bool = False,
        vocab: dict[str, int] | None = None,
        max_samples: int | None = None,
        stats: MotionStats | None = None,
        mirror_prob: float = 0.5,
        seed: int = 42,
        trans_stats_by_source: dict[str, MotionStats] | None = None,
    ):
        self.max_motion_length = max_motion_length
        self.max_text_length = max_text_length
        self.aug_pipeline = None
        self.mirror_prob = mirror_prob if augment and split == "train" else 0.0
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        self.trans_stats_by_source = trans_stats_by_source

        if augment:
            self.aug_pipeline = AugmentationPipeline(max_length=max_motion_length)

        self.samples, cache_stats = load_or_build_cache(data_dir, INGEST_MAX_LENGTH, max_samples)
        self.motion_stats = stats if stats is not None else cache_stats

        rng_split = np.random.default_rng(seed)
        rng_split.shuffle(self.samples)  # type: ignore[arg-type]
        n = len(self.samples)
        t_end = int(n * 0.8)   # 80 % train
        v_end = int(n * 0.9)   # 10 % val, 10 % test
        if split == "train":
            self.samples = self.samples[:t_end]
        elif split == "val":
            self.samples = self.samples[t_end:v_end]
        else:  # test -- held-out, never used for model selection
            self.samples = self.samples[v_end:]

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = build_vocab([s["text"] for s in self.samples])

        log.info("[MotionDataset] %s: %d samples, vocab=%d, augment=%s, mirror_p=%.2f",
                 split, len(self.samples), len(self.vocab), augment, self.mirror_prob)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.mirror_prob > 0.0 and self.rng.random() < self.mirror_prob:
            s = {**s, "motion": mirror_flip(s["motion"]), "text": mirror_flip_text(s["text"])}

        return encode_motion_sample(
            s, self.aug_pipeline,
            self.max_motion_length, self.max_text_length, self.vocab,
            stats=self.motion_stats,
            trans_stats_by_source=self.trans_stats_by_source,
        )
