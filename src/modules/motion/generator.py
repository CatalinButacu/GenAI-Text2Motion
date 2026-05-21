from __future__ import annotations

import logging

import numpy as np

from .blend import slerp_blend_frames
from .config import MotionConfig
from .models import MotionClip
from .ssm_model import SSMMotionModel

log = logging.getLogger(__name__)


class MotionGenerator:
    """Text-to-motion generator backed by the trained MotionSSM checkpoint."""

    def __init__(self, config: MotionConfig | None = None) -> None:
        cfg = config or MotionConfig()
        self.backend = SSMMotionModel(
            checkpoint_path=cfg.checkpoint_path,
            rvq_checkpoint_path=cfg.rvq_checkpoint_path,
        )
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p

    def generate(
        self,
        text: str,
        num_frames: int = 100,
        init_pose: np.ndarray | None = None,
        blend_frames: int = 10,
    ) -> MotionClip:
        log.info(
            "[MotionGen] generate(%r, n=%d, temp=%.2f, top_p=%.2f)",
            text, num_frames, self.temperature, self.top_p,
        )
        clip = self.backend.generate_from_text_tokens(
            text, num_frames, temperature=self.temperature, top_p=self.top_p
        )
        return blend_init_pose(clip, init_pose, blend_frames)


def blend_init_pose(
    clip: MotionClip, init_pose: np.ndarray | None, blend_frames: int
) -> MotionClip:
    if init_pose is None or clip.smplx_params is None:
        return clip

    n = min(blend_frames, len(clip.smplx_params))

    if n <= 0:
        return clip

    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a = np.broadcast_to(init_pose[None], (n, init_pose.shape[0])).copy()
    clip.smplx_params[:n] = slerp_blend_frames(a, clip.smplx_params[:n], alpha)

    return clip
