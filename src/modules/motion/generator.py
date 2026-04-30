from __future__ import annotations

import logging

import numpy as np

from .blend import slerpBlendFrames
from .config import MotionConfig
from .models import MotionClip
from .ssm_model import SSMMotionModel

log = logging.getLogger(__name__)


class MotionGenerator:
    """Text-to-motion generator backed by the trained MotionSSM checkpoint."""

    def __init__(self, config: MotionConfig | None = None) -> None:
        cfg = config or MotionConfig()
        self.backend = SSMMotionModel(
            checkpointPath=cfg.checkpointPath,
            rvqCheckpointPath=cfg.rvqCheckpointPath,
        )
        self.temperature = cfg.temperature
        self.topP = cfg.topP

    def generate(
        self,
        text: str,
        numFrames: int = 100,
        initPose: np.ndarray | None = None,
        blendFrames: int = 10,
    ) -> MotionClip:
        log.info(
            "[MotionGen] generate(%r, n=%d, temp=%.2f, top_p=%.2f)",
            text, numFrames, self.temperature, self.topP,
        )
        clip = self.backend.generateFromTextTokens(
            text, numFrames, temperature=self.temperature, topP=self.topP
        )
        return blendInitPose(clip, initPose, blendFrames)


def blendInitPose(
    clip: MotionClip, initPose: np.ndarray | None, blendFrames: int
) -> MotionClip:
    if initPose is None or clip.smplxParams is None:
        return clip

    n = min(blendFrames, len(clip.smplxParams))

    if n <= 0:
        return clip

    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a = np.broadcast_to(initPose[None], (n, initPose.shape[0])).copy()
    clip.smplxParams[:n] = slerpBlendFrames(a, clip.smplxParams[:n], alpha)

    return clip
