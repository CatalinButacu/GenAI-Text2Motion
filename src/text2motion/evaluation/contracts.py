from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import Field

from text2motion.generation.contracts import BackboneChoice, SamplingConfig
from text2motion.motion.contracts import DefaultConfigRequest, Split

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
    out_path: Path = Path("artifacts/outputs/streaming_bench.json")
    label: str = "streaming_bench"
    headroom: int = 8


############################################################
#       BenchmarkRequest
############################################################


class BenchmarkRequest(DefaultConfigRequest):
    device: str | None = Field(default=None, description="cuda or cpu; default from the config")
    horizons: tuple[int, ...] = Field(
        default=StreamingBenchmarkProtocol().horizons,
        description="stream lengths in tokens at which state size and latency are recorded",
    )
    report: Path | None = Field(
        default=None,
        description="output JSON file; default is streaming_bench.json in the outputs folder",
    )
    warmup: int = Field(
        default=StreamingBenchmarkProtocol().warmup,
        description="steps run before timing starts, so the first slow steps are not measured",
    )
    streams: tuple[int, ...] = Field(
        default=StreamingBenchmarkProtocol().streams,
        description="numbers of parallel streams to time; empty skips this test",
    )
    streams_horizon: int = Field(
        default=StreamingBenchmarkProtocol().streams_horizon,
        description="stream length used in the parallel-stream test",
    )


############################################################
#       EvaluateRequest
############################################################


class EvaluateRequest(DefaultConfigRequest):
    backbone: BackboneChoice = Field(
        default=BackboneChoice.BOTH,
        description="which generator to score (transformer, mamba, both); both uses one protocol",
    )
    split: Split = Field(
        default=GenerationEvaluationProtocol().split,
        description="data split (val or test); score test one time only, for a selected checkpoint",
    )
    max_clips: int = Field(
        default=GenerationEvaluationProtocol().max_clips,
        description="cap on clips; FID is biased upward when the count is small",
    )
    temperature: float = Field(
        default=SamplingConfig().temperature,
        description="sampling temperature; higher gives more varied and less accurate motion",
    )
    top_p: float = Field(
        default=SamplingConfig().top_p,
        description="nucleus threshold; lower removes unlikely tokens and reduces variety",
    )
    cfg_scale: float = Field(
        default=SamplingConfig().cfg_scale,
        description="guidance scale; 1 is no guidance, the thesis protocol uses 6",
    )
    length_mode: GenerationLengthPolicy = Field(
        default=GenerationLengthPolicy.FIXED,
        description="fixed uses the ground-truth length; end lets the model stop at its END token",
    )
    mm_clips: int = Field(
        default=GenerationEvaluationProtocol().mm_clips,
        description="captions used for the multimodality metric",
    )
    mm_repeats: int = Field(
        default=GenerationEvaluationProtocol().mm_repeats,
        description="motions generated for each of those captions",
    )
    bootstrap: int = Field(
        default=GenerationEvaluationProtocol().bootstrap,
        description="clip resamples for a bootstrap FID interval of generation run 0",
    )
    reps: int = Field(
        default=GenerationEvaluationProtocol().reps,
        description="generation runs; each run samples all clips again; the report gives mean and 95% interval",
    )
    eval_batch_size: int = Field(
        default=GenerationEvaluationProtocol().batch_size,
        description="clips generated together; changes speed, not the result",
    )
    ckpt: Path | None = Field(
        default=None,
        description="generator checkpoint path; used only when one backbone is selected",
    )
    tokenizer_ckpt: Path | None = Field(
        default=None, description="frozen tokenizer checkpoint; default from the config"
    )
    text_encoder_ckpt: Path | None = Field(
        default=None,
        description="checkpoint to read the text encoder from, when the generator file has none",
    )
    device: str | None = Field(default=None, description="cuda or cpu; default from the config")
    seed: int | None = Field(default=None, description="random seed; default from the config")
