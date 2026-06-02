from pathlib import Path

from src.modules.motion.models import MotionClip

from .config import RenderConfig
from .smplx_render import render_smplx2_video, view_smplx_interactive

__all__ = ["RenderConfig", "render_clip_to_file", "view_clip"]


def render_clip_to_file(clip: MotionClip, path: str, cfg: RenderConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    render_smplx2_video(
        clip.smplx_params,
        path,
        fps=cfg.fps,
        betas=clip.betas,
        gender=cfg.gender,
        width=cfg.width,
        height=cfg.height,
        input_coord_system=clip.coord_system,
        num_betas=cfg.num_betas,
    )


def view_clip(clip: MotionClip, cfg: RenderConfig) -> None:
    view_smplx_interactive(
        clip.smplx_params,
        fps=cfg.fps,
        betas=clip.betas,
        gender=cfg.gender,
        width=cfg.width,
        height=cfg.height,
        input_coord_system=clip.coord_system,
        num_betas=cfg.num_betas,
    )
