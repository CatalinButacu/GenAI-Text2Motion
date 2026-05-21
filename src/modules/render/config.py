from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RenderConfig:
    fps: int = 30
    gender: str = "neutral"
    width: int = 1280
    height: int = 720
    # Coordinate system of incoming SMPL-X params. HumanML3D motion is already
    # Y-up (matches aitviewer); raw AMASS / AMASS-trained model output is Z-up
    # and needs the canonical -90deg X / 180deg Y rotation to stand upright.
    input_coord_system: Literal["yup", "zup"] = "yup"
