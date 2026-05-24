from __future__ import annotations

import logging
from pathlib import Path

from src.utils.mem_profile import tracemalloc_snapshot

from .config import RenderConfig
from .smplx_render import render_smplx2_video, view_smplx_interactive

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


def view_interactive(motion_clips: dict, config: RenderConfig | None = None) -> None:
    """Open an interactive aitviewer window for the first actor's motion clip."""
    if not motion_clips:
        raise RuntimeError("[M6] no motion clips to view")

    cfg = config or RenderConfig()
    clip = next(iter(motion_clips.values()))
    view_smplx_interactive(
        clip.smplx_params,
        fps=cfg.fps,
        betas=clip.betas,
        gender=cfg.gender,
        width=cfg.width,
        height=cfg.height,
        input_coord_system=cfg.input_coord_system,
    )


__all__ = ["invoke", "view_interactive", "RenderConfig"]
