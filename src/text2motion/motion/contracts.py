from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from text2motion.motion.representation import FPS, JOINTS, MotionSource, RepresentationTrack

############################################################
#       Split
############################################################


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


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
#       MotionDataConfig
############################################################


@dataclass(frozen=True)
class MotionDataConfig:
    track: RepresentationTrack = RepresentationTrack.HML3D_263
    sources: tuple[MotionSource, ...] = (MotionSource.HUMANML3D,)
    mirror_augment: bool = True
    max_motion_len: int = 196
    min_motion_len: int = 40


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
