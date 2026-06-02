from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.plane import ChessboardPlane
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from scipy.spatial.transform import Rotation

from src.shared.constants import (
    BACKGROUND_COLOR,
    CAMERA_POSITION,
    CAMERA_TARGET,
    FLOOR_DIVISIONS,
    FLOOR_PLANE,
    FLOOR_PRIMARY,
    FLOOR_SECONDARY,
    FLOOR_SIZE,
    R_IDENTITY,
    R_ZUP_TO_YUP,
    SKIN_COLOR,
    SMPLX,
)
from src.shared.constants import (
    SMPLX_DIR as SHARED_SMPLX_DIR,
)

log = logging.getLogger(__name__)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SMPLX_DIR = os.path.join(ROOT, SHARED_SMPLX_DIR)

# Applied at import: SMPLLayer reads C.smplx_models in its constructor.
C.update_conf({"smplx_models": SMPLX_DIR})  # type: ignore[union-attr]

RENDERER: HeadlessRenderer | None = None
RENDERER_KEY: tuple[int, int] | None = None


def pick_rotation(input_coord_system: str) -> Rotation:
    if input_coord_system == "zup":
        return Rotation.from_matrix(R_ZUP_TO_YUP)

    if input_coord_system == "yup":
        return Rotation.from_matrix(R_IDENTITY)
    raise ValueError(f"unknown input_coord_system: {input_coord_system!r}")


def get_renderer(width: int, height: int) -> HeadlessRenderer:
    """Cache the HeadlessRenderer -- rebuilding one per call reboots the OpenGL context."""
    global RENDERER, RENDERER_KEY
    key = (width, height)

    if RENDERER is None or RENDERER_KEY != key:
        C.update_conf({"window_width": width, "window_height": height})  # type: ignore[union-attr]
        RENDERER = HeadlessRenderer()
        RENDERER_KEY = key

    return RENDERER


def reset_scene(renderer: HeadlessRenderer) -> None:
    """Clear SMPLSequence + floor nodes so successive renders don't stack meshes."""
    if renderer.scene is None:
        return

    for node in list(renderer.scene.nodes):  # type: ignore[union-attr]
        if isinstance(node, (SMPLSequence, ChessboardPlane)):
            renderer.scene.remove(node)  # type: ignore[union-attr]


def to_yup_coords(
    smplx_params: np.ndarray, input_coord_system: str
) -> tuple[np.ndarray, np.ndarray]:
    """Convert root orientation and translation to Y-up (aitviewer's world frame)."""
    rot = pick_rotation(input_coord_system)
    rot_mat = rot.as_matrix().astype(np.float32)
    trans = (smplx_params[:, SMPLX.transl_slice] @ rot_mat.T).astype(np.float32)
    root_orient = (
        (rot * Rotation.from_rotvec(smplx_params[:, SMPLX.root_orient_slice]))
        .as_rotvec()
        .astype(np.float32)
    )

    return root_orient, trans


def smplx_params2_sequence(
    smplx_params: np.ndarray,
    num_betas: int,
    betas: np.ndarray | None = None,
    gender: str = "neutral",
    color: tuple = SKIN_COLOR,
    input_coord_system: str = "yup",
) -> SMPLSequence:
    """Build an aitviewer SMPLSequence from a raw SMPL-X parameter array (T x 168)."""
    root_orient, trans = to_yup_coords(smplx_params, input_coord_system)
    body = smplx_params[:, SMPLX.body_pose_slice]
    lhand = smplx_params[:, SMPLX.lhand_pose_slice]
    rhand = smplx_params[:, SMPLX.rhand_pose_slice]

    betas = (
        np.zeros(num_betas, dtype=np.float32)
        if betas is None
        else np.asarray(betas, dtype=np.float32)[:num_betas]
    )
    smpl_layer = SMPLLayer(model_type="smplx", gender=gender, num_betas=len(betas), device=C.device)

    return SMPLSequence(
        poses_body=body,
        smpl_layer=smpl_layer,
        poses_root=root_orient,
        betas=betas,
        trans=trans,
        poses_left_hand=lhand,
        poses_right_hand=rhand,
        device=C.device,
        color=color,
    )


def configure_scene(viewer: Any, seq: SMPLSequence, fps: int) -> None:
    """Configure the aitviewer scene with floor, camera, and SMPL sequence."""
    viewer.scene.fps = fps
    viewer.playback_fps = fps
    viewer.scene.background_color = BACKGROUND_COLOR

    if viewer.scene.floor is not None:
        viewer.scene.remove(viewer.scene.floor)

    floor = ChessboardPlane(
        FLOOR_SIZE, FLOOR_DIVISIONS, FLOOR_PRIMARY, FLOOR_SECONDARY, FLOOR_PLANE
    )
    viewer.scene.floor = floor
    viewer.scene.add(floor)
    viewer.scene.add(seq)
    cam = viewer.scene.camera

    if cam is not None:
        cam.position = CAMERA_POSITION
        cam.target = CAMERA_TARGET


def render_smplx2_video(
    smplx_params: np.ndarray,
    output_path: str,
    fps: int,
    betas: np.ndarray | None,
    gender: str,
    width: int,
    height: int,
    input_coord_system: str,
    num_betas: int,
) -> None:
    """Render a Tx168 SMPL-X parameter sequence to an MP4 via aitviewer headless rendering."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seq = smplx_params2_sequence(
        smplx_params,
        num_betas=num_betas,
        betas=betas,
        gender=gender,
        input_coord_system=input_coord_system,
    )
    log.info("SMPLSequence: %d frames, gender=%s", len(smplx_params), gender)

    renderer = get_renderer(width=width, height=height)
    reset_scene(renderer)
    configure_scene(renderer, seq, fps)
    renderer.save_video(video_dir=str(out), output_fps=fps)

    log.info("video -> %s", output_path)


def view_smplx_interactive(
    smplx_params: np.ndarray,
    fps: int,
    betas: np.ndarray | None,
    gender: str,
    width: int,
    height: int,
    input_coord_system: str,
    num_betas: int,
    title: str = "Motion Preview",
) -> None:
    """Open an interactive aitviewer window to preview the motion."""
    C.update_conf({"window_width": width, "window_height": height})  # type: ignore[union-attr]

    seq = smplx_params2_sequence(
        smplx_params,
        num_betas=num_betas,
        betas=betas,
        gender=gender,
        input_coord_system=input_coord_system,
    )
    log.info("opening interactive viewer: %d frames @ %d fps", len(smplx_params), fps)

    v = Viewer(title=title, size=(width, height))
    configure_scene(v, seq, fps)
    v.run()
