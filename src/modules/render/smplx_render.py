from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.plane import ChessboardPlane
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from scipy.spatial.transform import Rotation

# SMPL-X parameters from HumanML3D are already Y-up (matches aitviewer).
# AMASS-native data needs Z-up→Y-up. The active rotation is selected per-render
# via RenderConfig.input_coord_system; pick_rotation() returns the right one.
R_ZUP_TO_YUP = Rotation.from_euler("xy", [-90, 180], degrees=True)
R_IDENTITY = Rotation.identity()


def pick_rotation(input_coord_system: str) -> Rotation:
    if input_coord_system == "zup":
        return R_ZUP_TO_YUP

    if input_coord_system == "yup":
        return R_IDENTITY
    raise ValueError(f"unknown input_coord_system: {input_coord_system!r}")

log = logging.getLogger(__name__)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SMPLX_DIR = os.path.join(ROOT, "data", "arctic", "unpack", "models")

# Apply the local model path immediately at import time so that SMPLLayer
# (which reads C.smplx_models in its constructor) always sees the right path,
# regardless of call order inside render_smplx2_video.
C.update_conf({"smplx_models": SMPLX_DIR})  # type: ignore[union-attr]

RENDERER: HeadlessRenderer | None = None
RENDERER_KEY: tuple[int, int] | None = None


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


def to_yup_coords(smplx_params: np.ndarray,
                input_coord_system: str = "yup") -> tuple[np.ndarray, np.ndarray]:
    """Convert root orientation and translation to Y-up (aitviewer's world frame)."""
    rot = pick_rotation(input_coord_system)
    rot_mat = rot.as_matrix().astype(np.float32)
    trans = (smplx_params[:, 3:6] @ rot_mat.T).astype(np.float32)
    root_orient = (
        (rot * Rotation.from_rotvec(smplx_params[:, 0:3])).as_rotvec().astype(np.float32)
    )

    return root_orient, trans


def smplx_params2_sequence(
    smplx_params: np.ndarray,
    betas: np.ndarray | None = None,
    gender: str = "neutral",
    color: tuple = (0.72, 0.60, 0.52, 1.0),
    input_coord_system: str = "yup",
) -> SMPLSequence:
    """Build an aitviewer SMPLSequence from a raw SMPL-X parameter array (T x 168)."""
    root_orient, trans = to_yup_coords(smplx_params, input_coord_system)
    body = smplx_params[:, 6:69]
    lhand = smplx_params[:, 69:114]
    rhand = smplx_params[:, 114:159]

    betas = (
        np.zeros(10, dtype=np.float32)
        if betas is None
        else np.asarray(betas, dtype=np.float32)[:10]
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


def configure_renderer(renderer: HeadlessRenderer, seq: SMPLSequence, fps: int) -> None:
    """Set scene properties, swap the default floor, place the camera, and add the sequence."""
    renderer.scene.fps = fps  # type: ignore[union-attr]
    renderer.playback_fps = fps
    renderer.scene.background_color = [0.85, 0.87, 0.90, 1.0]  # type: ignore[union-attr]

    if renderer.scene.floor is not None:  # type: ignore[union-attr]
        renderer.scene.remove(renderer.scene.floor)  # type: ignore[union-attr]

    floor = ChessboardPlane(100.0, 200, (0.82, 0.83, 0.84, 1.0), (0.80, 0.81, 0.82, 1.0), "xz")
    renderer.scene.floor = floor  # type: ignore[union-attr]
    renderer.scene.add(floor)  # type: ignore[union-attr]
    renderer.scene.add(seq)  # type: ignore[union-attr]
    cam = renderer.scene.camera  # type: ignore[union-attr]

    if cam is not None:
        cam.position = np.array([0.0, 1.5, 4.5])
        cam.target = np.array([0.0, 1.0, 0.0])


def render_smplx2_video(
    smplx_params: np.ndarray,
    output_path: str,
    fps: int = 30,
    betas: np.ndarray | None = None,
    gender: str = "neutral",
    width: int = 1280,
    height: int = 720,
    input_coord_system: str = "yup",
) -> None:
    """Render a Tx168 SMPL-X parameter sequence to an MP4 via aitviewer headless rendering."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seq = smplx_params2_sequence(smplx_params, betas=betas, gender=gender,
                               input_coord_system=input_coord_system)
    log.info("[M6] SMPLSequence: %d frames, gender=%s", len(smplx_params), gender)

    renderer = get_renderer(width=width, height=height)
    reset_scene(renderer)
    configure_renderer(renderer, seq, fps)
    renderer.save_video(video_dir=str(out), output_fps=fps)

    log.info("[M6] video -> %s", output_path)


def view_smplx_interactive(
    smplx_params: np.ndarray,
    title: str = "Motion Preview",
    fps: int = 30,
    betas: np.ndarray | None = None,
    gender: str = "neutral",
    width: int = 1280,
    height: int = 720,
    input_coord_system: str = "yup",
) -> None:
    """Open an interactive aitviewer window to preview the motion in real-time.

    Controls: space=play/pause, left/right arrows=scrub, scroll=zoom,
    left-drag=orbit, right-drag=pan.
    """
    C.update_conf({"window_width": width, "window_height": height})  # type: ignore[union-attr]

    seq = smplx_params2_sequence(smplx_params, betas=betas, gender=gender,
                                 input_coord_system=input_coord_system)
    log.info("[M6] opening interactive viewer: %d frames @ %d fps", len(smplx_params), fps)

    v = Viewer(title=title, size=(width, height))
    v.scene.fps = fps
    v.playback_fps = fps
    v.scene.background_color = [0.85, 0.87, 0.90, 1.0]

    if v.scene.floor is not None:
        v.scene.remove(v.scene.floor)

    floor = ChessboardPlane(100.0, 200, (0.82, 0.83, 0.84, 1.0), (0.80, 0.81, 0.82, 1.0), "xz")
    v.scene.floor = floor
    v.scene.add(floor)
    v.scene.add(seq)

    cam = v.scene.camera
    if cam is not None:
        cam.position = np.array([0.0, 1.5, 4.5])
        cam.target = np.array([0.0, 1.0, 0.0])

    v.run()
