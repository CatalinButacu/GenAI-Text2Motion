from __future__ import annotations

import logging

from src.shared.constants import MOTION_FPS
from src.utils.mem_profile import tracemallocSnapshot

from .clip_ops import generateActionClips, sequenceClips
from .config import MotionConfig
from .generator import MotionGenerator
from .models import MotionClip
from .ssm_model import SSMMotionModel

log = logging.getLogger(__name__)

GENERATOR: MotionGenerator | None = None
GENERATOR_CKPT: str | None = None


def getGenerator(cfg: MotionConfig) -> MotionGenerator:
    """Cache a MotionGenerator keyed on checkpoint path.

    Only the heavy weight-loading step is cached. Cheap per-call sampling
    params (temperature, topP) are refreshed on every call so config changes
    across pipeline.run() invocations within the same process are honored.
    """
    global GENERATOR, GENERATOR_CKPT

    if GENERATOR is None or GENERATOR_CKPT != cfg.checkpointPath:
        GENERATOR = MotionGenerator(cfg)
        GENERATOR_CKPT = cfg.checkpointPath
    else:
        GENERATOR.temperature = cfg.temperature
        GENERATOR.topP = cfg.topP

    return GENERATOR


def invoke(planned, config: MotionConfig | None = None) -> dict[str, MotionClip]:
    """M4: generate SMPL-X pose sequences for each actor via the trained MotionSSM."""
    cfg = config or MotionConfig()
    generator = getGenerator(cfg)

    duration = getattr(planned, "duration", 5.0)
    totalFrames = int(duration * MOTION_FPS)
    log.info("[M4] target: %d frames (%.1fs x %dfps)", totalFrames, duration, MOTION_FPS)

    with tracemallocSnapshot("M4 motion"):
        actionClips = generateActionClips(planned, totalFrames, generator, cfg)

        if not actionClips:
            raise RuntimeError("[M4] no motion clips generated -- prompt has no recognised actions")

        clips = sequenceClips(actionClips, cfg.blendFrames)

    return clips


__all__ = [
    "invoke",
    "MotionConfig",
    "MotionClip",
    "MotionGenerator",
    "SSMMotionModel",
    "MOTION_FPS",
]
