from __future__ import annotations

import logging

import numpy as np
from aitviewer.scene.camera import ViewerCamera

from text2motion.studio.contracts import ViewerHost

logger = logging.getLogger(__name__)


class StreamCamera:
    def __init__(self, host: ViewerHost, state) -> None:
        self.host = host
        self.state = state

    def center_view(self, node) -> None:
        bounds = node.current_bounds
        if not np.isfinite(bounds).all():
            logger.info(f"[warn] skipped centring on {node.name}: non-finite bounds")
            return
        self.host.center_view_on_node(node)

    def current_center(self) -> np.ndarray | None:
        frame = self.host.scene.current_frame_id
        nodes = self.state.nodes
        body_verts = self.state.stream_buffers.body_verts
        joints = self.state.stream_buffers.accum_joints
        body_frames = 0 if body_verts is None else len(body_verts)
        body_is_shown = nodes.body is not None and (nodes.motion is None or frame < body_frames)
        if body_is_shown and body_verts is not None:
            return body_verts[min(frame, len(body_verts) - 1)].mean(axis=0)
        if body_is_shown or nodes.motion is None or joints is None:
            return None
        return joints[min(frame, len(joints) - 1)].mean(axis=0)

    def update_follow_cam(self) -> None:
        if not self.state.follow_cam:
            return

        camera = self.host.scene.camera
        if not isinstance(camera, ViewerCamera):
            return

        center = self.current_center()
        if center is None:
            return

        if self.state.nodes.follow_center is None:
            offset = camera.position - camera.target
            camera.target = center
            camera.position = center + offset
        else:
            delta = center - self.state.nodes.follow_center
            camera.position = camera.position + delta
            camera.target = camera.target + delta

        self.state.nodes.follow_center = center
