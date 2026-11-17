from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from text2motion.motion.representation import MotionSource, RepresentationTrack

GRAVITY_M_PER_S2 = 9.81
SEED = 2026
AMASS_LABEL_FPS = 120
MOTIONX_FPS = 30
MOTIONX_POINTS = 144
MOTIONX_SUBSETS = ("animation", "haa500", "humman", "idea400", "kungfu", "music", "perform")
STEP0_AMASS_SUBSETS = frozenset({"GRAB", "BMLrub", "HUMAN4D"})
CLIP_NAME_SEPARATOR = "__"
UNSAFE_FILE_CHARACTERS = r'[<>:"/\\|?* ]'
MIN_UPRIGHT_SHARE = 0.5
MIN_CLIP_SECONDS = 2.0
HEAD_JOINT = 15
FOOT_JOINTS = (10, 11)
FEET = ((7, 10), (8, 11))
SEGMENTS = (
    (12, 15, 6.94),
    (0, 12, 43.46),
    (16, 18, 2.71),
    (17, 19, 2.71),
    (18, 20, 1.62),
    (19, 21, 1.62),
    (20, 20, 0.61),
    (21, 21, 0.61),
    (1, 4, 14.16),
    (2, 5, 14.16),
    (4, 7, 4.33),
    (5, 8, 4.33),
    (7, 10, 1.37),
    (8, 11, 1.37),
)
CONTACT_HEIGHT_M = (
    0.15,
    0.15,
    0.15,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.05,
    0.05,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.08,
    0.08,
)


def clip_name(*parts: str) -> str:
    return re.sub(UNSAFE_FILE_CHARACTERS, "_", CLIP_NAME_SEPARATOR.join(parts))


def subset_of(name: str, source: str) -> str:
    return name.split(CLIP_NAME_SEPARATOR)[0] if CLIP_NAME_SEPARATOR in name else source


############################################################
#       NewtonLimits
############################################################
@dataclass(frozen=True)
class NewtonLimits:
    hover_fraction_of_g: float = 0.5
    quasi_static_m_per_s2: float = 1.0
    support_margin_m: float = 0.10
    min_flight_seconds: float = 0.15
    min_unbalanced_seconds: float = 0.25
    reject_violation_share: float = 0.5


############################################################
#       QualityLimits
############################################################
@dataclass(frozen=True)
class QualityLimits:
    contact_height_m: float = 0.05
    skid_m_per_s: float = 0.5
    float_tolerance_m: float = 0.005
    yaw_percentile: float = 99.0
    trees: int = 100


############################################################
#       DatasetPaths
############################################################
@dataclass(frozen=True)
class DatasetPaths:
    repo: Path
    amass_raw: Path

    @property
    def hml3d_official(self) -> Path:
        return self.repo / "data" / "HumanML3D_official"

    @property
    def extra(self) -> Path:
        return self.repo / "data" / "extra_motion"

    @property
    def motionx(self) -> Path:
        return self.extra / "motionx_pp"

    @property
    def smplx_models(self) -> Path:
        return self.repo / "data" / "smplx_models"

    @property
    def notes(self) -> Path:
        return self.repo / "doc" / "notes"

    def hml3d_at(self, fps: int) -> Path:
        return self.repo / "data" / f"HumanML3D_{fps}fps"

    def joints(self, fps: int, source: str) -> Path:
        return self.extra / f"joints{fps}" / source

    def features(self, fps: int, source: str) -> Path:
        return self.extra / f"features{fps}" / source


############################################################
#       CaptionWeighting
############################################################
class CaptionWeighting(StrEnum):
    UNIFORM = "uniform"
    RARE_VERB = "rare_verb"


############################################################
#       ClipWeighting
############################################################
class ClipWeighting(StrEnum):
    UNIFORM = "uniform"
    MOTION_CLASS = "motion_class"
    UNIFORM_REPLACEMENT = "uniform_replacement"
    VERB = "verb"


############################################################
#       MotionDataConfig
############################################################
@dataclass(frozen=True)
class MotionDataConfig:
    track: RepresentationTrack = RepresentationTrack.HML3D_263
    sources: tuple[MotionSource, ...] = (MotionSource.HUMANML3D,)
    mirror_augment: bool = True
    caption_weighting: CaptionWeighting = CaptionWeighting.UNIFORM
    clip_weighting: ClipWeighting = ClipWeighting.UNIFORM
    motion_classes: Path | None = None
    max_motion_len: int = 196
    min_motion_len: int = 40

    def __post_init__(self) -> None:
        if not 0 < self.min_motion_len <= self.max_motion_len:
            raise ValueError(
                f"motion lengths need 0 < min_motion_len <= max_motion_len, got "
                f"{self.min_motion_len} and {self.max_motion_len}"
            )
        if not self.sources:
            raise ValueError("the data config needs at least one motion source")
        if self.clip_weighting is ClipWeighting.MOTION_CLASS and self.motion_classes is None:
            raise ValueError("motion_class clip weighting needs the motion_classes verb map")
        if (
            self.clip_weighting is ClipWeighting.VERB
            and self.caption_weighting is CaptionWeighting.RARE_VERB
        ):
            raise ValueError("verb clip weighting picks its own caption; use uniform captions")
