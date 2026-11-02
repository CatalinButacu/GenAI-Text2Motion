from __future__ import annotations

import logging

import numpy as np
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import ViewerCamera

from text2motion.motion.representation import kinematic_bones
from text2motion.studio.contracts import ViewerHost

logger = logging.getLogger(__name__)


def create_skeleton_sequence_node(joints: np.ndarray):
    from aitviewer.renderables.skeletons import Skeletons

    return Skeletons(joint_positions=joints, joint_connections=kinematic_bones())


def append_skeleton_sequence_frames(node, joints: np.ndarray) -> None:
    node.joint_positions = np.append(node.joint_positions, joints, axis=0)
    node.spheres.add_frames(joints)
    node.lines.add_frames(joints[:, node.skeleton].reshape(len(joints), -1, 3))


############################################################
#       StudioSceneNodes
############################################################
class StudioSceneNodes:
    def __init__(self, scene) -> None:
        self.scene = scene
        self.rest = None
        self.motion = None
        self.body = None
        self.follow_center: np.ndarray | None = None

    def __drop(self, node):
        if node is None:
            return
        if self.scene.selected_object is node:
            self.scene.select(None)
        self.scene.remove(node)

    def clear_rest(self) -> None:
        self.rest = self.__drop(self.rest)

    def clear_motion(self) -> None:
        self.motion = self.__drop(self.motion)

    def clear_body(self) -> None:
        self.body = self.__drop(self.body)

    def clear_all(self) -> None:
        self.clear_motion()
        self.clear_rest()
        self.clear_body()

    def show_rest(self, node) -> None:
        self.clear_rest()
        self.rest = node
        self.scene.add(node)

    def show_motion(self, node) -> None:
        self.clear_motion()
        self.motion = node
        self.scene.add(node)

    def show_body(self, node) -> None:
        self.body = node
        self.scene.add(node)

    @property
    def focus(self):
        return self.body or self.motion or self.rest

    @property
    def has_generated_motion(self) -> bool:
        return self.body is not None or self.motion is not None


############################################################
#       StreamSceneUpdater
############################################################
class StreamSceneUpdater:
    def __init__(self, host: ViewerHost, state) -> None:
        self.host = host
        self.state = state

    def consume(self) -> None:
        chunks, body_chunks, remesh = self.state.stream_buffers.drain()

        if chunks:
            self.append_chunks(chunks)
        if body_chunks:
            self.__append_body_chunks(body_chunks)
        if remesh is not None:
            self.__apply_remesh(*remesh)
        if self.state.stream_buffers.need_reset_view:
            self.state.stream_buffers.need_reset_view = False
            node = self.state.nodes.focus
            if node is not None:
                self.__center_view(node)

    def hold_playhead_at_fitted_body(self) -> None:
        buffers = self.state.stream_buffers
        motion = self.state.nodes.motion
        if not buffers.fitting or not self.state.show_skin or motion is None:
            return
        if self.state.nodes.body is None or buffers.body_frames == 0:
            self.host.scene.current_frame_id = 0
            return
        if buffers.body_frames >= len(motion.joint_positions):
            return
        if self.host.scene.current_frame_id >= buffers.body_frames:
            self.host.scene.current_frame_id = buffers.body_frames - 1

    def sync_display_nodes(self) -> None:
        body = self.state.nodes.body
        motion = self.state.nodes.motion
        if body is None:
            if motion is not None:
                motion.enabled = True
            return
        if not self.state.show_skin:
            body.enabled = False
            if motion is not None:
                motion.enabled = True
            return
        body_frames = (
            0
            if self.state.stream_buffers.body_verts is None
            else len(self.state.stream_buffers.body_verts)
        )
        on_body = self.host.scene.current_frame_id < body_frames
        body.enabled = on_body
        if motion is not None:
            motion.enabled = not on_body

    def __center_view(self, node) -> None:
        bounds = node.current_bounds
        if not np.isfinite(bounds).all():
            logger.info(f"[warn] skipped centring on {node.name}: non-finite bounds")
            return
        self.host.center_view_on_node(node)

    def append_chunks(self, chunks: list[np.ndarray]) -> None:
        new = np.concatenate(chunks, axis=0)
        if not self.state.stream_buffers.started:
            self.state.nodes.clear_motion()
            self.state.stream_buffers.accum_joints = None
            self.state.stream_buffers.trimmed_frames = 0
        self.state.stream_buffers.accum_joints = (
            new
            if self.state.stream_buffers.accum_joints is None
            else np.concatenate([self.state.stream_buffers.accum_joints, new], axis=0)
        )
        if self.state.nodes.motion is None:
            motion = create_skeleton_sequence_node(new)
            motion.name = "Motion (streaming)"
            self.state.nodes.show_motion(motion)
        else:
            append_skeleton_sequence_frames(self.state.nodes.motion, new)
        logger.info(
            f"[DISP] skeleton node added: accum={self.state.stream_buffers.accum_joints.shape} "
            f"scene_frames={self.host.scene.n_frames} started={self.state.stream_buffers.started}"
        )
        self.__trim_to_window()
        if not self.state.stream_buffers.started:
            self.state.stream_buffers.started = True
            self.state.nodes.clear_body()
            self.state.nodes.clear_rest()
            self.state.nodes.follow_center = None
            self.host.scene.current_frame_id = 0
            self.host.toggle_animation(True)
            self.state.stream_buffers.need_reset_view = True

    def __trim_to_window(self) -> None:
        window = self.state.window_frames
        motion = self.state.nodes.motion
        if window <= 0 or motion is None:
            return
        excess = len(motion.joint_positions) - window
        body = self.state.nodes.body
        if body is not None:
            excess = min(excess, len(body.vertices) - 1)
        excess -= excess % self.state.upsample
        if excess <= 0:
            return
        dropped = list(range(excess))
        self.state.stream_buffers.trimmed_frames += excess
        motion.joint_positions = motion.joint_positions[excess:]
        motion.spheres.remove_frames(dropped)
        motion.lines.remove_frames(dropped)
        if body is not None:
            body.remove_frames(dropped)
            self.state.stream_buffers.body_verts = body.vertices
            with self.state.stream_buffers.lock:
                self.state.smplx_fit.drop_first_frames(excess)
        kept = self.state.stream_buffers.accum_joints[excess:]
        self.state.stream_buffers.accum_joints = kept
        self.state.stream_buffers.keep_last_source_frames(len(kept) // self.state.upsample + 1)
        self.host.scene.current_frame_id = max(0, self.host.scene.current_frame_id - excess)

    def __append_body_chunks(self, chunks: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for verts, faces in chunks:
            if self.state.nodes.body is None:
                self.state.nodes.show_body(
                    Meshes(verts, faces, color=self.state.skin_color, name="Avatar (SMPL-X)")
                )
                self.state.stream_buffers.body_verts = verts
            else:
                self.state.nodes.body.add_frames(verts)
                self.state.stream_buffers.body_verts = self.state.nodes.body.vertices
        motion = self.state.nodes.motion
        if motion is None:
            return
        lag = len(motion.joint_positions) - len(self.state.stream_buffers.body_verts)
        if lag == self.state.stream_buffers.last_body_lag:
            return
        self.state.stream_buffers.last_body_lag = lag
        logger.info(
            f"[DISP] body frames={len(self.state.stream_buffers.body_verts)} "
            f"skeleton frames={len(motion.joint_positions)} lag={lag}"
        )

    def __apply_remesh(self, verts: np.ndarray, faces: np.ndarray, is_rest: bool) -> None:
        if is_rest:
            if self.state.nodes.has_generated_motion:
                return
            self.state.nodes.show_rest(
                Meshes(verts, faces, color=self.state.skin_color, name="Avatar (rest)")
            )
            return
        frame = self.host.scene.current_frame_id
        self.state.nodes.clear_all()
        self.state.nodes.show_body(
            Meshes(verts, faces, color=self.state.skin_color, name="Avatar (SMPL-X)")
        )
        self.state.stream_buffers.body_verts = verts
        self.host.scene.current_frame_id = min(frame, len(verts) - 1)

    def __current_center(self) -> np.ndarray | None:
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

        center = self.__current_center()
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
