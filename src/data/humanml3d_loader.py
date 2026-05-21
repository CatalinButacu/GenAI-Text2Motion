from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.amass import AMASSLoader
from src.data.augmentation import (
    AugmentationPipeline,
    canonicalize_root,
    mirror_flip,
    mirror_flip_text,
    quality_filter,
)
from src.data.motion_normalize import MotionStats, compute_motion_stats, normalize
from src.shared.tokenizer import build_vocab
from src.shared.tokenizer import tokenize as tok

from .humanml3d import (
    DEFAULT_DIR,
    HumanML3DLoader,
    build_norm_map,
    load_motion_item,
    normalize_amass_path,
    parse_index_csv,
    preload_smplx,
)

__all__ = ["HumanML3DLoader", "normalize_amass_path", "HumanML3DMotionDataset"]

log = logging.getLogger(__name__)


class HumanML3DMotionDataset(Dataset):
    def __init__(self, data_dir=DEFAULT_DIR, split="train", max_motion_length=200,
                 max_text_length=64, augment=False, vocab=None, preload=True,
                 stats: MotionStats | None = None, seed: int = 42,
                 mirror_prob: float = 0.5, amass_dir: str = "data/AMASS",
                 min_frames: int = 30, apply_quality_filter: bool = True) -> None:
        self.dir = Path(data_dir)
        self.amass_dir = Path(amass_dir)
        self.max_motion, self.max_text = max_motion_length, max_text_length
        self.aug_pipeline = AugmentationPipeline(max_length=max_motion_length) if augment else None
        self.split = split
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        self.mirror_prob = mirror_prob if augment and split == "train" else 0.0
        self.min_frames = min_frames
        self.apply_quality_filter = apply_quality_filter
        self.samples = self.load_samples(split)

        all_texts_flat = [t for s in self.samples for t in s["texts"]]
        self.vocab = vocab if vocab is not None else build_vocab(all_texts_flat)

        self.cache = self.preload(preload)
        self.motion_stats = stats if stats is not None else self.compute_stats()

    def load_samples(self, split: str) -> list:
        loader = HumanML3DLoader(str(self.dir))
        split_ids = set(loader.load_split(split))
        all_texts = {item["clip_id"]: item["texts"] for item in loader.load_texts()}
        idx_f = self.dir / "index.csv"

        if not idx_f.exists():
            raise FileNotFoundError(f"{idx_f} missing.")

        amass = AMASSLoader(str(self.amass_dir))
        norm_map = build_norm_map(amass.discover_files(), amass.data_dir)

        return parse_index_csv(idx_f, split_ids, all_texts, norm_map)

    def preload(self, preload: bool) -> dict | None:
        if not preload:
            return None

        log.info("[HumanML3D] preloading %d samples...", len(self.samples))
        cache, bad = preload_smplx(self.samples, self.amass_dir)

        if bad:
            bad_s = set(bad)
            self.samples = [s for s in self.samples if s["clip_id"] not in bad_s]

        # Apply the same hygiene the unified loader does so train/eval see the
        # same effective dataset regardless of --data-source. min_frames drops
        # sub-second subclips; quality_filter rejects static / kinematically
        # implausible motion. Both checks match unified_load.load_humanml3d.
        keep: list[dict] = []
        n_short = n_quality = 0
        filtered_cache: dict = {}

        for s in self.samples:
            clip_id = s["clip_id"]
            m = cache.get(clip_id)

            if m is None:
                continue

            if m.shape[0] < self.min_frames:
                n_short += 1
                continue

            if self.apply_quality_filter and not quality_filter(m, 30.0):
                n_quality += 1
                continue
            filtered_cache[clip_id] = canonicalize_root(m)
            keep.append(s)
        self.samples = keep
        log.info(
            "[HumanML3D] %s: kept %d  (short<%d: %d, qualityRejected: %d, preloadBad: %d)",
            self.split, len(keep), self.min_frames, n_short, n_quality, len(bad or []),
        )

        return filtered_cache

    def compute_stats(self) -> MotionStats:
        """Compute normalisation stats over this split's loaded motion clips."""
        if self.cache is None or not self.cache:
            raise RuntimeError(
                "HumanML3DMotionDataset: cannot compute stats without preloaded cache"
            )
        samples_for_stats = [{"motion": m} for m in self.cache.values()]
        s = compute_motion_stats(samples_for_stats)
        log.info("[HumanML3D] motion stats: mean|.|=%.4f std|.|=%.4f",
                 float(np.abs(s.mean).mean()), float(np.abs(s.std).mean()))

        return s

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        motion = load_motion_item(s, self.cache, self.amass_dir)
        text = (
            s["texts"][self.rng.integers(0, len(s["texts"]))]
            if self.split == "train" else s["text"]
        )

        if self.mirror_prob > 0.0 and self.rng.random() < self.mirror_prob:
            motion = mirror_flip(motion)
            text = mirror_flip_text(text)

        if self.aug_pipeline is not None:
            motion = self.aug_pipeline(motion)
        motion = normalize(motion, self.motion_stats)
        T = min(motion.shape[0], self.max_motion)
        motion = np.pad(motion[:T], ((0, self.max_motion - T), (0, 0)))
        mask = np.zeros(self.max_motion, dtype=np.float32)
        mask[:T] = 1.0
        token_ids = tok(text, self.vocab, max_len=self.max_text)

        return {"token_ids": torch.tensor(token_ids, dtype=torch.long),
                "texts": text,
                "motion": torch.tensor(motion, dtype=torch.float32),
                "motion_mask": torch.tensor(mask, dtype=torch.float32),
                "length": T}
