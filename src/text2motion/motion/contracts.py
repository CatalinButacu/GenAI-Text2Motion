from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from text2motion.motion.representation import FPS, JOINTS, MotionSource, RepresentationTrack


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class PreparationStage(StrEnum):
    AMASS = "amass"
    INDEX = "index"
    FEATURE = "feature"
    STATS = "stats"
    ALL = "all"


@dataclass(frozen=True)
class MotionDataConfig:
    track: RepresentationTrack = RepresentationTrack.HML3D_263
    sources: tuple[MotionSource, ...] = (MotionSource.HUMANML3D,)
    mirror_augment: bool = True
    max_motion_len: int = 196
    min_motion_len: int = 40


@dataclass(frozen=True)
class TextAnnotation:
    caption: str
    tokens: list[str]
    start_time: float
    end_time: float


@dataclass(frozen=True)
class PreparationRequest:
    out_dir: Path
    amass_dir: Path | None = None
    smplx_models: Path | None = None
    index_csv: Path | None = None
    device: str = "cpu"
    fps: int = FPS
    joint_count: int = JOINTS
