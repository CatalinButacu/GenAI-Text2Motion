from __future__ import annotations

import logging
from pathlib import Path

from src.utils.mem_profile import tracemalloc_snapshot

from .config import RenderConfig
from .smplx_render import render_smplx2_video

log = logging.getLogger(__name__)


def invoke(motion_clips: dict, output_path: str, config: RenderConfig | None = None) -> str:
    """M6: render the first actor's SMPL-X clip to MP4 via aitviewer headless."""
    if not motion_clips:
        raise RuntimeError("[M6] no motion clips to render")

    cfg = config or RenderConfig()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clip = next(iter(motion_clips.values()))

    with tracemalloc_snapshot("M6 rendering"):
        render_smplx2_video(
            clip.smplx_params,
            output_path,
            fps=cfg.fps,
            betas=clip.betas,
            gender=cfg.gender,
            width=cfg.width,
            height=cfg.height,
            input_coord_system=cfg.input_coord_system,
        )

    log.info("[M6] video saved -> %s (%d fps)", output_path, cfg.fps)

    return output_path


__all__ = ["invoke", "RenderConfig"]
