from __future__ import annotations

import logging

from src.shared.constants import MOTION_FPS
from src.utils.mem_profile import tracemalloc_snapshot

from .clip_ops import generate_action_clips, sequence_clips
from .config import MotionConfig
from .generator import MotionGenerator
from .models import MotionClip
from .ssm_model import SSMMotionModel

log = logging.getLogger(__name__)

GENERATOR: MotionGenerator | None = None
GENERATOR_CKPT: str | None = None


def get_generator(cfg: MotionConfig) -> MotionGenerator:
    """Cache a MotionGenerator keyed on checkpoint path.

    Only the heavy weight-loading step is cached. Cheap per-call sampling
    params (temperature, top_p) are refreshed on every call so config changes
    across pipeline.run() invocations within the same process are honored.
    """
    global GENERATOR, GENERATOR_CKPT

    if GENERATOR is None or GENERATOR_CKPT != cfg.checkpoint_path:
        GENERATOR = MotionGenerator(cfg)
        GENERATOR_CKPT = cfg.checkpoint_path
    else:
        GENERATOR.temperature = cfg.temperature
        GENERATOR.top_p = cfg.top_p
        GENERATOR.cfg_scale = cfg.cfg_scale

    return GENERATOR


def invoke(planned, config: MotionConfig | None = None) -> dict[str, MotionClip]:
    """M4: generate SMPL-X pose sequences for each actor via the trained MotionSSM."""
    cfg = config or MotionConfig()
    generator = get_generator(cfg)

    duration = getattr(planned, "duration", 5.0)
    total_frames = int(duration * MOTION_FPS)
    log.info("[M4] target: %d frames (%.1fs x %dfps)", total_frames, duration, MOTION_FPS)

    with tracemalloc_snapshot("M4 motion"):
        action_clips = generate_action_clips(planned, total_frames, generator, cfg)

        if not action_clips:
            raise RuntimeError("[M4] no motion clips generated -- prompt has no recognised actions")

        clips = sequence_clips(action_clips, cfg.blend_frames)

    return clips


__all__ = [
    "invoke",
    "MotionConfig",
    "MotionClip",
    "MotionGenerator",
    "SSMMotionModel",
    "MOTION_FPS",
]
