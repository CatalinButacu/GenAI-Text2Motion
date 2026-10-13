from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from text2motion.motion.contracts import CommandRequest, CorpusLayout, Split
from text2motion.motion.representation import DYNAMICS_TERMS, FK_TERMS, GEO_TERMS


class Backbone(StrEnum):
    TRANSFORMER = "transformer"
    MAMBA = "mamba"


class BackboneChoice(StrEnum):
    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    BOTH = "both"


class MixedPrecisionMode(StrEnum):
    OFF = "off"
    BF16 = "bf16"


class LossWeighting(StrEnum):
    FIXED = "fixed"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True)
class OverfitCriteria:
    min_token_accuracy: float = 0.99
    max_ce: float = 0.1
    max_total_ratio: float = 0.05

    def accepts(self, token_accuracy: float, ce: float, total_ratio: float) -> bool:
        return (
            token_accuracy >= self.min_token_accuracy
            and ce <= self.max_ce
            and total_ratio <= self.max_total_ratio
        )


DEFAULT_OVERFIT_CRITERIA = OverfitCriteria()


@dataclass(frozen=True)
class GeneratorConfig:
    backbone: Backbone = Backbone.MAMBA
    d_model: int = 512
    n_layers: int = 8
    mamba_n_layers: int = 15
    d_text: int = 512
    num_codebooks: int = 6
    codebook_size: int = 1000
    max_seq_len: int = 96
    dropout: float = 0.1
    text_prefix_len: int = 1
    use_end_token: bool = False
    use_kernel: bool = False
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 32
    n_heads: int = 8


@dataclass(frozen=True)
class TextEncoderConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    out_dim: int = 512
    max_length: int = 77
    unfreeze_last_n: int = 1
    unfreeze_projection: bool = True
    prefix_len: int = 1


@dataclass(frozen=True)
class LossWeights:
    root: float = 0.3
    ric: float = 0.5
    rot6d: float = 0.5
    vel: float = 0.3
    foot: float = 0.1
    fk_self: float = 0.0
    fk_gt: float = 0.0
    foot_skate: float = 0.0
    accel: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def active_terms(self) -> tuple[str, ...]:
        weights = self.as_dict()
        return tuple(name for name in (*GEO_TERMS, *FK_TERMS, *DYNAMICS_TERMS) if weights[name] > 0)

    @property
    def fk_enabled(self) -> bool:
        return self.fk_self > 0 or self.fk_gt > 0

    @property
    def dynamics_enabled(self) -> bool:
        return self.foot_skate > 0 or self.accel > 0

    @property
    def needs_positions(self) -> bool:
        return self.fk_enabled or self.dynamics_enabled


@dataclass(frozen=True)
class GeneratorTrainingConfig:
    lr: float = 2e-4
    text_encoder_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_min_ratio: float = 0.01
    ema_decay: float = 0.999
    cfg_dropout: float = 0.1
    pkeep: float = 0.8
    amp: MixedPrecisionMode = MixedPrecisionMode.OFF
    grad_accum: int = 1
    grad_clip: float = 1.0
    decay_groups: bool = True
    loss_weighting: LossWeighting = LossWeighting.FIXED
    loss_weights: LossWeights = field(default_factory=LossWeights)


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.9
    cfg_scale: float = 1.0
    stop_at_end: bool = False


@dataclass(frozen=True)
class StudioSamplingConfig(SamplingConfig):
    cfg_scale: float = 6.0


@dataclass(frozen=True)
class OverfitGateRequest:
    batch_size: int = 4
    steps: int = 1000
    lr: float = 3e-3


@dataclass(frozen=True)
class TextToMotionGenerationRequest:
    prompt: str
    token_steps: int
    sampling: SamplingConfig = SamplingConfig()
    chunk_tokens: int = 1


@dataclass(frozen=True)
class GeneratorTrainingRequest:
    epochs: int = 60
    batch_size: int = 64
    num_workers: int = 0
    grad_accum: int = 1
    eval_every: int = 5
    eval_split: Split = Split.VALIDATION
    max_eval_clips: int = 600
    eval_batch_size: int = 32
    temperature: float = 1.0
    cfg_scale: float = 1.0
    checkpoint_name: str | None = None
    resume: bool = False
    fresh_selection: bool = False
    patience: int = 0


@dataclass(frozen=True)
class GeneratorPretrainingRequest:
    token_pack: Path
    epochs: int = 30
    batch_size: int = 64
    grad_accum: int = 1
    num_workers: int = 0
    val_fraction: float = 0.05
    out_path: Path | None = None
    resume: bool = False


class SanityOverfitRequest(CommandRequest):
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
        default=OverfitGateRequest().batch_size,
        description="clips in the one batch that the model must memorise",
    )
    steps: int = Field(
        default=OverfitGateRequest().steps,
        description="maximum optimizer steps before the gate fails",
    )
    lr: float = Field(
        default=OverfitGateRequest().lr,
        description="learning rate for the gate only; high on purpose, to memorise fast",
    )


class TrainGeneratorRequest(CommandRequest):
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
    eval_split: Split = Field(
        default=GeneratorTrainingRequest().eval_split,
        description="split for checkpoint selection; use val, test is for the final score only",
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
        description="with --resume: forget the earlier best FID, so selection restarts in this run",
    )
    resume_run_dir: Path | None = Field(
        default=None,
        description="existing run folder to continue the log in; keeps one folder for one training",
    )
    patience: int = Field(
        default=GeneratorTrainingRequest().patience,
        description="stop after this many validations without a better FID; 0 never stops early",
    )


class PretrainRequest(CommandRequest):
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
