from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from text2motion.motion.representation import FPS, JOINTS


############################################################
#       Split
############################################################
class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


type MetricSink = Callable[[dict[str, object]], None]
type Evaluator = Callable[[Split], dict[str, float]]


def ignore_metrics(metrics: dict[str, object]) -> None:
    return None


############################################################
#       RunOutputs
############################################################
@dataclass(frozen=True)
class RunOutputs:
    run_dir: Path | None = None
    on_metrics: MetricSink = ignore_metrics
    failure_dump_path: Path | None = None


NO_RUN_OUTPUTS = RunOutputs()


############################################################
#       PreparationStage
############################################################
class PreparationStage(StrEnum):
    AMASS = "amass"
    INDEX = "index"
    FEATURE = "feature"
    STATS = "stats"
    ALL = "all"


############################################################
#       TextAnnotation
############################################################
@dataclass(frozen=True)
class TextAnnotation:
    caption: str
    tokens: list[str]
    start_time: float
    end_time: float


############################################################
#       PreparationRequest
############################################################
@dataclass(frozen=True)
class PreparationRequest:
    out_dir: Path
    amass_dir: Path | None = None
    smplx_models: Path | None = None
    index_csv: Path | None = None
    device: str = "cpu"
    fps: int = FPS
    joint_count: int = JOINTS


############################################################
#       CorpusLayout
############################################################
@dataclass(frozen=True)
class CorpusLayout:
    official_dir: Path = Path("data/HumanML3D_official")
    regen_dir: Path = Path("data/HumanML3D_263")
    amass_corpus_dir: Path = Path("data/AMASS_263")
    amass_features_dir: Path = Path("data/AMASS_263/new_joint_vecs")
    amass_token_pack: Path = Path("data/amass_tokens.npz")
    pretraining_token_pack: Path = Path("data/amass_tokens_fsq8x1024.npz")


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


SMPLX_MODEL_TYPE = "smplx"


############################################################
#       Gender
############################################################
class Gender(StrEnum):
    NEUTRAL = "neutral"
    MALE = "male"
    FEMALE = "female"


############################################################
#       MotionClip
############################################################
@dataclass(frozen=True)
class MotionClip:
    features: torch.Tensor
    frame_count: int
    caption: str | None = None
    clip_id: str | None = None


############################################################
#       MotionBatch
############################################################
@dataclass(frozen=True)
class MotionBatch:
    features: torch.Tensor
    lengths: torch.Tensor
    captions: list[str]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: str | torch.device) -> MotionBatch:
        return MotionBatch(
            features=self.features.to(device),
            lengths=self.lengths.to(device),
            captions=self.captions,
        )


############################################################
#       MotionTokens
############################################################
@dataclass(frozen=True)
class MotionTokens:
    indices: torch.Tensor
    frame_count: int

    @property
    def token_count(self) -> int:
        return int(self.indices.shape[0])


############################################################
#       MotionChunk
############################################################
@dataclass(frozen=True)
class MotionChunk:
    motion: MotionClip
    index: int
    is_final: bool


############################################################
#       StreamedChunk
############################################################
@dataclass(frozen=True)
class StreamedChunk:
    joints: np.ndarray
    features: np.ndarray
    yaw: float


############################################################
#       GeneratedMotion
############################################################
@dataclass(frozen=True)
class GeneratedMotion:
    prompt: str
    tokens: MotionTokens
    motion: MotionClip
