from __future__ import annotations

from pathlib import Path

from pydantic import Field

from text2motion.app.requests.base import (
    DefaultConfigRequest,
)
from text2motion.evaluation.contracts import (
    GenerationEvaluationProtocol,
    GenerationLengthPolicy,
    StreamingBenchmarkProtocol,
)
from text2motion.generation.contracts import (
    BackboneChoice,
    SamplingConfig,
)
from text2motion.motion.contracts import Split


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
    clip_list: Path | None = Field(
        default=None,
        description="file with one clip id per line; only these clips of the split are scored",
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
