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
from scipy.spatial.transform import Rotation

# Z-up (SMPL-X world) -> Y-up (aitviewer world) coordinate transform
R_ZUP_TO_YUP = Rotation.from_euler("xy", [-90, 180], degrees=True)
R_MAT = R_ZUP_TO_YUP.as_matrix().astype(np.float32)

log = logging.getLogger(__name__)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SMPLX_DIR = os.path.join(ROOT, "data", "arctic", "unpack", "models")

RENDERER: HeadlessRenderer | None = None
RENDERER_KEY: tuple[int, int] | None = None


def getRenderer(width: int, height: int) -> HeadlessRenderer:
    """Cache the HeadlessRenderer -- rebuilding one per call reboots the OpenGL context."""
    global RENDERER, RENDERER_KEY
    key = (width, height)

    if RENDERER is None or RENDERER_KEY != key:
        C.update_conf({"smplx_models": SMPLX_DIR, "window_width": width, "window_height": height})  # type: ignore[union-attr]
        RENDERER = HeadlessRenderer()
        RENDERER_KEY = key

    return RENDERER


def resetScene(renderer: HeadlessRenderer) -> None:
    """Clear SMPLSequence + floor nodes so successive renders don't stack meshes."""
    if renderer.scene is None:
        return

    for node in list(renderer.scene.nodes):  # type: ignore[union-attr]
        if isinstance(node, (SMPLSequence, ChessboardPlane)):
            renderer.scene.remove(node)  # type: ignore[union-attr]


def toYupCoords(smplxParams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert root orientation and translation from Z-up (SMPL-X) to Y-up (aitviewer)."""
    trans = (smplxParams[:, 3:6] @ R_MAT.T).astype(np.float32)
    rootOrient = (
        (R_ZUP_TO_YUP * Rotation.from_rotvec(smplxParams[:, 0:3])).as_rotvec().astype(np.float32)
    )

    return rootOrient, trans


def smplxParams2Sequence(
    smplxParams: np.ndarray,
    betas: np.ndarray | None = None,
    gender: str = "neutral",
    color: tuple = (0.72, 0.60, 0.52, 1.0),
) -> SMPLSequence:
    """Build an aitviewer SMPLSequence from a raw SMPL-X parameter array (T x 168)."""
    rootOrient, trans = toYupCoords(smplxParams)
    body = smplxParams[:, 6:69]
    lhand = smplxParams[:, 69:114]
    rhand = smplxParams[:, 114:159]

    betas = (
        np.zeros(10, dtype=np.float32)
        if betas is None
        else np.asarray(betas, dtype=np.float32)[:10]
    )
    smplLayer = SMPLLayer(model_type="smplx", gender=gender, num_betas=len(betas), device=C.device)

    return SMPLSequence(
        poses_body=body,
        smpl_layer=smplLayer,
        poses_root=rootOrient,
        betas=betas,
        trans=trans,
        poses_left_hand=lhand,
        poses_right_hand=rhand,
        device=C.device,
        color=color,
    )


def configureRenderer(renderer: HeadlessRenderer, seq: SMPLSequence, fps: int) -> None:
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


def renderSmplx2Video(
    smplxParams: np.ndarray,
    outputPath: str,
    fps: int = 30,
    betas: np.ndarray | None = None,
    gender: str = "neutral",
    width: int = 1280,
    height: int = 720,
) -> None:
    """Render a Tx168 SMPL-X parameter sequence to an MP4 via aitviewer headless rendering."""
    out = Path(outputPath)
    out.parent.mkdir(parents=True, exist_ok=True)

    seq = smplxParams2Sequence(smplxParams, betas=betas, gender=gender)
    log.info("[M6] SMPLSequence: %d frames, gender=%s", len(smplxParams), gender)

    renderer = getRenderer(width=width, height=height)
    resetScene(renderer)
    configureRenderer(renderer, seq, fps)
    renderer.save_video(video_dir=str(out), output_fps=fps)

    log.info("[M6] video -> %s", outputPath)
