from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from text2motion.evaluation.guo.checkpoint import (
    load_frozen_guo_embedding_encoders,
    load_guo_motion_normalization_stats,
)
from text2motion.evaluation.guo.vocabulary import (
    GuoWordPosVectorizer,
    create_guo_text_input_builder,
)
from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM
from text2motion.motion.storage import SPLIT_FILES, base_clip_ids, listed_clip_ids

OFFICIAL_POPULATION_SPLITS = frozenset({Split.TEST})


def resolve_annotation_dir(motion_dataset_dir: Path, annotation_dir: Path | None) -> Path:
    return Path(annotation_dir) if annotation_dir else motion_dataset_dir / "texts"


def build_guo_text_builder(vocab_dir: Path) -> Callable:
    return create_guo_text_input_builder(GuoWordPosVectorizer(str(vocab_dir), "our_vab"))


############################################################
#       GuoEvaluationResources
############################################################
@dataclass(frozen=True)
class GuoEvaluationResources:
    device: str
    motion_dataset_dir: Path
    annotation_dir: Path
    dataset_motion_scaler: MotionScaler
    guo_motion_mean: np.ndarray
    guo_motion_std: np.ndarray
    motion_embedder: Any
    text_embedder: Any
    build_guo_text_inputs: Callable
    tokenizer_downsample_factor: int

    @classmethod
    def from_frozen_guo_assets(
        cls,
        motion_dataset_dir: Path,
        eval_stats_dir: Path,
        eval_matcher: Path,
        vocab_dir: Path,
        tokenizer_downsample_factor: int,
        annotation_dir: Path | None = None,
        device: str = "cpu",
    ) -> GuoEvaluationResources:
        motion_dataset_dir = Path(motion_dataset_dir)
        guo_motion_mean, guo_motion_std = load_guo_motion_normalization_stats(eval_stats_dir)
        motion_embedder, text_embedder = load_frozen_guo_embedding_encoders(
            eval_matcher, device=device
        )

        return cls(
            device=device,
            motion_dataset_dir=motion_dataset_dir,
            annotation_dir=resolve_annotation_dir(motion_dataset_dir, annotation_dir),
            dataset_motion_scaler=MotionScaler.load(motion_dataset_dir, dim=DIM),
            guo_motion_mean=guo_motion_mean,
            guo_motion_std=guo_motion_std,
            motion_embedder=motion_embedder,
            text_embedder=text_embedder,
            build_guo_text_inputs=build_guo_text_builder(vocab_dir),
            tokenizer_downsample_factor=tokenizer_downsample_factor,
        )

    def clip_ids(self, split: Split | str) -> list[str]:
        if Split(split) in OFFICIAL_POPULATION_SPLITS:
            return listed_clip_ids(self.motion_dataset_dir / SPLIT_FILES[Split(split)])
        return base_clip_ids(self.motion_dataset_dir, split)

    def normalize(self, feats):
        return self.dataset_motion_scaler.normalize(feats)

    def denormalize(self, feats):
        return self.dataset_motion_scaler.denormalize(feats)
