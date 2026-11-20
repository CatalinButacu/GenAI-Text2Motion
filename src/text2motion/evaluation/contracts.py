from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from text2motion.generation.contracts import SamplingConfig
from text2motion.motion.contracts import Split

R_PRECISION_POOL = 32
DIVERSITY_PAIRS = 300
GUO_MIN_FRAMES = 40
GUO_MAX_FRAMES = 200


############################################################
#       GenerationLengthPolicy
############################################################
class GenerationLengthPolicy(StrEnum):
    FIXED = "fixed"
    END_TOKEN = "end"


############################################################
#       GenerationEvaluationProtocol
############################################################
@dataclass(frozen=True)
class GenerationEvaluationProtocol:
    split: Split = Split.TEST
    max_clips: int = 100000
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    mm_clips: int = 100
    mm_repeats: int = 30
    bootstrap: int = 200
    reps: int = 20
    batch_size: int = 32
    clip_list: tuple[str, ...] | None = None


############################################################
#       GenerationEvaluationReport
############################################################
@dataclass(frozen=True)
class GenerationEvaluationReport:
    clips: int
    runs: int
    fid: float
    r_precision: tuple[float, float, float]
    diversity: float
    multimodality: float
    matching_distance: float
    fid_ci95: float
    r_precision_ci95: tuple[float, float, float]
    diversity_ci95: float
    matching_distance_ci95: float
    fid_bootstrap_lo: float = float("nan")
    fid_bootstrap_hi: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        record = asdict(self)
        top1, top2, top3 = self.r_precision
        top1_ci, top2_ci, top3_ci = self.r_precision_ci95
        record.pop("r_precision")
        record.pop("r_precision_ci95")
        record.update({"r_top1": top1, "r_top2": top2, "r_top3": top3})
        record.update({"r_top1_ci95": top1_ci, "r_top2_ci95": top2_ci, "r_top3_ci95": top3_ci})
        return record


############################################################
#       EvaluationSample
############################################################
@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    reference: np.ndarray
    token_len: int
    captions: list[str]
    caption_tokens: list[list[str]]


############################################################
#       GenerationEvaluationSet
############################################################
@dataclass(frozen=True)
class GenerationEvaluationSet:
    sample_ids: list[str]
    reference: list[np.ndarray]
    generated: list[np.ndarray]
    token_lists: list[list[list[str]]]
    captions: list[str]
    requested: int
    dropped: dict[str, int]


############################################################
#       StreamingBenchmarkProtocol
############################################################
@dataclass(frozen=True)
class StreamingBenchmarkProtocol:
    horizons: tuple[int, ...] = (64, 128, 256, 512, 1024)
    warmup: int = 32
    streams: tuple[int, ...] = ()
    streams_horizon: int = 256
    out_path: Path = Path("runs/outputs/streaming_bench.json")
    label: str = "streaming_bench"
    headroom: int = 8
