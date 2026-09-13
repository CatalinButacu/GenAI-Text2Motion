from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from text2motion.motion.contracts import Split
from text2motion.motion.representation import FK_TERMS, GEO_TERMS


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

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def active_terms(self) -> tuple[str, ...]:
        weights = self.as_dict()
        return tuple(name for name in (*GEO_TERMS, *FK_TERMS) if weights[name] > 0)

    @property
    def fk_enabled(self) -> bool:
        return self.fk_self > 0 or self.fk_gt > 0


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
class TextToMotionGenerationRequest:
    prompt: str
    token_steps: int
    sampling: SamplingConfig = SamplingConfig()
    chunk_tokens: int = 1


@dataclass(frozen=True)
class GeneratorTrainingRequest:
    epochs: int = 60
    batch_size: int = 64
    grad_accum: int = 1
    eval_every: int = 5
    eval_split: Split = Split.VALIDATION
    max_eval_clips: int = 600
    eval_batch_size: int = 32
    temperature: float = 1.0
    cfg_scale: float = 1.0
    checkpoint_name: str | None = None
    resume: bool = False
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
