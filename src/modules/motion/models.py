from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.shared.constants import CONSTS


class MotionSource(Enum):
    RETRIEVAL = "retrieval"
    SSM = "ssm"
    MOMASK = "momask"
    SEQUENCED = "sequenced"


@dataclass(slots=True)
class MotionClip:
    action: str
    smplx_params: np.ndarray
    fps: int = CONSTS.runtime.motion_fps
    source: MotionSource = MotionSource.SSM
    coord_system: str = "yup"
    raw_joints: np.ndarray | None = None  # shape (T, 22, 3) Z-up metres
    betas: np.ndarray | None = None  # SMPL-X shape coefficients (16,)

    @property
    def duration(self) -> float:
        return len(self.smplx_params) / self.fps

    @property
    def num_frames(self) -> int:
        return len(self.smplx_params)
