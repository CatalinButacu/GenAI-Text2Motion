from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from text2motion.generation.contracts import SamplingConfig
from text2motion.motion.contracts import Split

R_PRECISION_POOL = 32
DIVERSITY_PAIRS = 300


class GenerationLengthPolicy(StrEnum):
    FIXED = "fixed"
    END_TOKEN = "end"


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


@dataclass(frozen=True)
class GenerationEvaluationReport:
    clips: int
    fid: float
    r_precision: tuple[float, float, float]
    diversity: float
    multimodality: float
    matching_distance: float
    r_top1_std: float = 0.0
    fid_ci_lo: float = float("nan")
    fid_ci_hi: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        record = asdict(self)
        top1, top2, top3 = self.r_precision
        record.pop("r_precision")
        record.update({"r_top1": top1, "r_top2": top2, "r_top3": top3})
        return record


@dataclass(frozen=True)
class GenerationEvaluationSet:
    reference: list[np.ndarray]
    generated: list[np.ndarray]
    token_lists: list[list[list[str]]]
    captions: list[str]
    requested: int
    dropped: dict[str, int]


@dataclass(frozen=True)
class StreamingBenchmarkProtocol:
    horizons: tuple[int, ...] = (64, 128, 256, 512, 1024)
    warmup: int = 32
    streams: tuple[int, ...] = ()
    streams_horizon: int = 256
    out_path: Path = Path("outputs/streaming_bench.json")
    label: str = "streaming_bench"
    headroom: int = 8
