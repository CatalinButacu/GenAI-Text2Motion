from __future__ import annotations

import logging
from pathlib import Path

from src.utils.mem_profile import tracemallocSnapshot

from .config import RenderConfig
from .smplx_render import renderSmplx2Video

log = logging.getLogger(__name__)


def invoke(motionClips: dict, outputPath: str, config: RenderConfig | None = None) -> str:
    """M6: render the first actor's SMPL-X clip to MP4 via aitviewer headless."""
    if not motionClips:
        raise RuntimeError("[M6] no motion clips to render")

    cfg = config or RenderConfig()
    Path(outputPath).parent.mkdir(parents=True, exist_ok=True)
    clip = next(iter(motionClips.values()))

    with tracemallocSnapshot("M6 rendering"):
        renderSmplx2Video(
            clip.smplxParams,
            outputPath,
            fps=cfg.fps,
            betas=clip.betas,
            gender=cfg.gender,
            width=cfg.width,
            height=cfg.height,
        )

    log.info("[M6] video saved -> %s (%d fps)", outputPath, cfg.fps)

    return outputPath


__all__ = ["invoke", "RenderConfig"]
