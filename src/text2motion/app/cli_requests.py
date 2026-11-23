from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.evaluation.contracts import (
    GenerationEvaluationProtocol,
    GenerationLengthPolicy,
    StreamingBenchmarkProtocol,
)
from text2motion.generation.contracts import (
    Backbone,
    BackboneChoice,
    GeneratorPretrainingRequest,
    GeneratorTrainingRequest,
    SamplingConfig,
    StudioSamplingConfig,
)
from text2motion.motion.contracts import CorpusLayout, PreparationStage, Split
from text2motion.motion.representation import FPS
from text2motion.tokenization.contracts import TokenizerKind

DEFAULT_SERVICE_CONFIG = Path("configs/service.yml")
DEFAULT_CONFIG = Path("configs/default.yaml")
CONFIG_HELP = "YAML file with model, data and path settings; it selects the experiment"


############################################################
#       CommandRequest
############################################################
class CommandRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    config: Path = Field(description=CONFIG_HELP)

    @field_validator("*", mode="before")
    @classmethod
    def member_from_name(cls, value: Any, info: ValidationInfo) -> Any:
        annotation = cls.model_fields[info.field_name or ""].annotation
        if not (isinstance(annotation, type) and issubclass(annotation, Enum)):
            return value
        if isinstance(value, str) and value in annotation.__members__:
            return annotation[value]
        return value


############################################################
#       DefaultConfigRequest
############################################################
class DefaultConfigRequest(CommandRequest):
    config: Path = Field(default=DEFAULT_CONFIG, description=CONFIG_HELP)


############################################################
#       ServiceAddressRequest
############################################################
class ServiceAddressRequest(DefaultConfigRequest):
    host: str | None = Field(default=None, description="service address; default from the config")
    port: int | None = Field(default=None, description="service TCP port; default from the config")
    service_config: Path = Field(
        default=DEFAULT_SERVICE_CONFIG,
        description="YAML file with the address and runtime settings of the service",
    )


############################################################
#       HealthRequest
############################################################
class HealthRequest(ServiceAddressRequest):
    pass


############################################################
#       ModelServiceRequest
############################################################
class ModelServiceRequest(ServiceAddressRequest):
    ckpt: Path | None = Field(
        default=None, description="generator checkpoint; default from the config"
    )
    tokenizer_ckpt: Path | None = Field(
        default=None,
        description="frozen tokenizer checkpoint; it must be the one the generator was trained on",
    )
    backbone: Backbone = Field(
        default=Backbone.TRANSFORMER,
        description="sequence mixer: mamba has a fixed-size state, transformer has a cache that grows",
    )


############################################################
#       ServeRequest
############################################################
class ServeRequest(ModelServiceRequest):
    idle_seconds: int | None = Field(
        default=None,
        description="the service stops after this many seconds without a request",
    )


############################################################
#       StudioRequest
############################################################
class StudioRequest(ModelServiceRequest):
    model_dir: Path | None = Field(
        default=None,
        description="SMPL-X model folder; without it the studio shows the skeleton only",
    )
    studio_config: Path = Field(
        default=Path("configs/studio.yml"),
        description="YAML file with display, fit and panel settings",
    )
    model: str | None = Field(
        default=None,
        description="key of a model in the model registry; it sets config, checkpoint and backbone",
    )
    steps: int = Field(
        default=0,
        description="token steps for each generation; one step is 4 frames; 0 uses the model maximum",
    )
    seconds: float = Field(
        default=0.0,
        description="motion length in seconds for each generation; above 0 it replaces --steps",
    )
    cfg_scale: float = Field(
        default=StudioSamplingConfig().cfg_scale,
        description="guidance scale; higher follows the text more closely and varies less",
    )
    temperature: float = Field(
        default=StudioSamplingConfig().temperature,
        description="sampling temperature; higher gives more varied motion",
    )
    top_p: float = Field(
        default=StudioSamplingConfig().top_p,
        description="nucleus threshold; lower removes unlikely tokens",
    )
    fps: int = Field(
        default=FPS, description="frame rate of the motion data, before display upsampling"
    )


############################################################
#       PrepareRequest
############################################################
class PrepareRequest(DefaultConfigRequest):
    stage: PreparationStage = Field(
        default=PreparationStage.ALL,
        description="run one preparation stage only (amass, index, feature, stats); all runs them in sequence",
    )
    corpus: bool = Field(
        default=False,
        description="build the unlabeled AMASS pretraining corpus and not the captioned data set",
    )
    adopt_pose_folder: bool = Field(
        default=False,
        description="stage amass: accept an existing pose folder without a manifest after a new "
        "extraction of a fixed sample matches the files on disk",
    )
    regen_dir: Path = Field(
        default=CorpusLayout().regen_dir,
        description="folder with the regenerated poses and joints that the corpus is built from",
    )
    corpus_dir: Path = Field(
        default=CorpusLayout().amass_corpus_dir,
        description="output folder of the AMASS 263-feature corpus",
    )
    min_frames: int = Field(
        default=MotionDataConfig().min_motion_len,
        description="clips shorter than this are dropped; a higher value gives fewer, longer clips",
    )
    limit: int = Field(
        default=0,
        description="convert only the first N sequences, for a quick test; 0 converts all",
    )
    source_release: Path | None = Field(
        default=None,
        description="folder of an official HumanML3D release to convert; skips all other stages",
    )
    converted_dir: Path = Field(
        default=CorpusLayout().official_dir,
        description="output folder for the converted official release",
    )


############################################################
#       SanityOverfitRequest
############################################################
class SanityOverfitRequest(CommandRequest):
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol run that is not citable; protocol runs use seed 2026",
    )
    backbone: Backbone = Field(
        description="sequence mixer: mamba has a fixed-size state, transformer has a cache that grows"
    )
    tokenizer_ckpt: Path = Field(
        description="frozen tokenizer checkpoint; it must be the one the generator was trained on"
    )
    overfit_gate: Path = Field(
        description="output gate file; train-generator refuses to start without it"
    )
    batch_size: int = Field(
        default=4,
        description="clips in the one batch that the model must memorise",
    )
    steps: int = Field(
        default=1000,
        description="maximum optimizer steps before the gate fails",
    )
    lr: float = Field(
        default=3e-3,
        description="learning rate for the gate only; high on purpose, to memorise fast",
    )
    init_ckpt: Path | None = Field(
        default=None,
        description="pretrained weights to start the gate from, as the run that uses the gate does",
    )


############################################################
#       TrainGeneratorRequest
############################################################
class TrainGeneratorRequest(CommandRequest):
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol run that is not citable; protocol runs use seed 2026",
    )
    backbone: Backbone = Field(
        description="sequence mixer: mamba has a fixed-size state, transformer has a cache that grows"
    )
    tokenizer_ckpt: Path | None = Field(
        default=None,
        description="frozen tokenizer checkpoint; it must be the one the generator was trained on",
    )
    overfit_gate: Path = Field(
        description="passed gate file from sanity-overfit for this config, tokenizer and backbone"
    )
    init_ckpt: Path | None = Field(
        default=None,
        description="pretrained weights to start from; without it training starts from random weights",
    )
    text_cache: bool = Field(
        default=True,
        description="reuse the frozen CLIP layer states of the train captions from a disk cache; the results are the same, the step is faster",
    )
    compile: bool = Field(
        default=True,
        description="compile the generator with torch.compile; about 7% faster for mamba, the results are not bit-identical to an eager run; without CUDA or Triton the run stays eager",
    )
    epochs: int = Field(
        default=GeneratorTrainingRequest().epochs,
        description="target epoch; the cosine schedule spans exactly this many epochs",
    )
    batch_size: int = Field(
        default=GeneratorTrainingRequest().batch_size,
        description="clips per step; keep it equal for both backbones",
    )
    num_workers: int = Field(
        default=GeneratorTrainingRequest().num_workers,
        description="data loader processes; more is faster, and each one needs RAM; 0 needs the least",
    )
    grad_accum: int = Field(
        default=GeneratorTrainingRequest().grad_accum,
        description="batches per optimizer step; effective batch is batch_size times this",
    )
    cfg_scale: float = Field(
        default=GeneratorTrainingRequest().cfg_scale,
        description="guidance scale for the validation samples; higher follows the text more closely",
    )
    eval_every: int = Field(
        default=GeneratorTrainingRequest().eval_every,
        description="epochs between validation runs; each run can select a new best checkpoint",
    )
    max_eval_clips: int = Field(
        default=GeneratorTrainingRequest().max_eval_clips,
        description="validation clips; fewer is faster and gives a noisier FID",
    )
    eval_batch_size: int = Field(
        default=GeneratorTrainingRequest().eval_batch_size,
        description="clips generated together in validation; changes speed, not the result",
    )
    temperature: float = Field(
        default=GeneratorTrainingRequest().temperature,
        description="sampling temperature for validation; higher gives more varied motion",
    )
    vocab_dir: Path | None = Field(
        default=None,
        description="word-vector folder of the evaluator; default from the config",
    )
    ckpt_name: str | None = Field(
        default=None,
        description="file name of the best checkpoint inside the checkpoint folder",
    )
    resume: bool = Field(
        default=False, description="continue from the saved last state of ckpt_name"
    )
    fresh_selection: bool = Field(
        default=False,
        description="with --resume: reset earlier best FID and validation-CE patience for this run",
    )
    resume_run_dir: Path | None = Field(
        default=None,
        description="existing run folder to continue the log in; keeps one folder for one training",
    )
    patience: int = Field(
        default=GeneratorTrainingRequest().patience,
        description="stop after this many validations without a better FID; 0 never stops early",
    )
    clean_ce: bool = Field(
        default=GeneratorTrainingRequest().clean_ce,
        description="after each epoch, log the teacher-forced CE on a fixed train subset and on validation",
    )
    ce_batches: int = Field(
        default=GeneratorTrainingRequest().ce_batches,
        description="batches of --eval_batch_size clips for each clean CE; 0 uses the full split",
    )
    val_ce_patience: int = Field(
        default=GeneratorTrainingRequest().val_ce_patience,
        ge=0,
        description="stop on this consecutive epoch without a lower clean validation CE; default 6 allows five retries; 0 disables",
    )
    stop_epoch: int = Field(
        default=GeneratorTrainingRequest().stop_epoch,
        ge=0,
        description="stop after this epoch while --epochs still sets the cosine length; 0 runs to --epochs",
    )
    select_on_val_ce: bool = Field(
        default=GeneratorTrainingRequest().select_on_val_ce,
        description="save the best bundle at the lowest clean validation CE, not at the lowest val FID",
    )
    save_r1_floor: float = Field(
        default=GeneratorTrainingRequest().save_r1_floor,
        ge=0,
        le=1,
        description="save the best-FID bundle only at an evaluation whose val R@1 is at least this value; an evaluation below it counts as no improvement; default 0 keeps every run unchanged",
    )
    accept_legacy_resume: bool = Field(
        default=False,
        description="with --resume: load an old resume file; its RNG and schedule state start fresh",
    )
    extend_schedule: bool = Field(
        default=False,
        description="with --resume: accept a new --epochs; the learning rate follows the new cosine",
    )


############################################################
#       PretrainRequest
############################################################
class PretrainRequest(CommandRequest):
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol run that is not citable; protocol runs use seed 2026",
    )
    backbone: Backbone = Field(
        description="sequence mixer: mamba has a fixed-size state, transformer has a cache that grows"
    )
    token_pack: Path = Field(
        default=CorpusLayout().pretraining_token_pack,
        description="token file from the tokenize command; the unlabeled motion to learn from",
    )
    epochs: int = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).epochs,
        description="passes over the token corpus; too many memorise it",
    )
    batch_size: int = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).batch_size,
        description="windows per step; keep it equal for both backbones",
    )
    grad_accum: int = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).grad_accum,
        description="batches per optimizer step; effective batch is batch_size times this",
    )
    num_workers: int = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).num_workers,
        description="data loader processes; 0 needs the least RAM",
    )
    val_fraction: float = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).val_fraction,
        description="share of clips held out, by clip; shows if the model memorises the corpus",
    )
    ckpt: Path | None = Field(
        default=None,
        description="output checkpoint path; default is generator_<backbone>_pretrained.pt",
    )
    resume: bool = Field(
        default=False, description="continue from the saved state of this checkpoint"
    )
    accept_legacy_resume: bool = Field(
        default=False,
        description="with --resume: load an old resume file; its RNG and schedule state start fresh",
    )
    extend_schedule: bool = Field(
        default=False,
        description="with --resume: accept a new --epochs; the learning rate follows the new cosine",
    )


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


############################################################
#       TokenizeRequest
############################################################
class TokenizeRequest(CommandRequest):
    features_dir: Path = Field(
        default=CorpusLayout().amass_features_dir,
        description="folder of 263-feature clips to encode",
    )
    tokenizer_ckpt: Path | None = Field(
        default=None,
        description="frozen tokenizer checkpoint; default from the config",
    )
    token_pack: Path = Field(
        default=CorpusLayout().amass_token_pack,
        description="output .npz file of tokens; pretraining reads this file",
    )
    segment_frames: int = Field(
        default=MotionDataConfig().max_motion_len,
        gt=0,
        description="window length in frames; clips shorter than one window are not encoded",
    )
    stride: int = Field(
        default=MotionDataConfig().max_motion_len,
        gt=0,
        description="step between windows; a smaller stride gives more, overlapping windows",
    )
    regen_dir: Path = Field(
        default=CorpusLayout().regen_dir,
        description="folder with pose_data; the heldout clips are found and excluded through it",
    )


############################################################
#       TrainTokenizerRequest
############################################################
class TrainTokenizerRequest(CommandRequest):
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol run that is not citable; protocol runs use seed 2026",
    )
    tokenizer: TokenizerKind = Field(
        description="quantizer family: fsq has no learned codebook, rvq is the baseline"
    )
    epochs: int = Field(default=50, gt=0, description="passes over the training set")
    window: int = Field(
        default=64,
        gt=0,
        description="training crop in frames; a longer crop gives more context and uses more memory",
    )
    batch_size: int = Field(default=128, gt=0, description="clips per step; limited by GPU memory")
    ema_decay: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="decay of the averaged weights that are evaluated; higher is smoother and slower",
    )
    eval_every: int = Field(
        default=5, gt=0, description="epochs between reconstruction evaluations"
    )
    max_eval_clips: int | None = Field(
        default=None,
        gt=0,
        description="cap on evaluation clips; a low cap is faster and gives a noisier FID",
    )
    max_steps: int | None = Field(
        default=None,
        gt=0,
        description="stop after this many optimizer steps, for equal-step comparisons; not with --resume",
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="data loader processes; 0 loads in the training process and needs the least RAM",
    )
    amass_share: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="share of AMASS clips in each epoch; 0 trains on HumanML3D only",
    )
    amass_features: Path | None = Field(
        default=None,
        description="folder with the AMASS 263 features; required when amass_share is above 0",
    )
    ckpt_name: str | None = Field(
        default=None,
        description="checkpoint file name inside the checkpoint folder; also names the run",
    )
    resume: bool = Field(
        default=False, description="continue from the saved state of this checkpoint"
    )
    accept_legacy_resume: bool = Field(
        default=False,
        description="with --resume: load an old resume file; its RNG state starts fresh",
    )
