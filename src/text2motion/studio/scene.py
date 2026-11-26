from __future__ import annotations

import logging

import numpy as np
from aitviewer.renderables.meshes import Meshes

from text2motion.motion.representation import kinematic_bones
from text2motion.studio.contracts import ViewerHost
from text2motion.studio.scene_support.camera import StreamCamera
from text2motion.studio.scene_support.clearer import SceneNodeClearer

logger = logging.getLogger(__name__)


def create_skeleton_sequence_node(joints: np.ndarray, radius: float = 0.01):
    from aitviewer.renderables.skeletons import Skeletons

    return Skeletons(joint_positions=joints, joint_connections=kinematic_bones(), radius=radius)


def append_skeleton_sequence_frames(node, joints: np.ndarray) -> None:
    node.joint_positions = np.append(node.joint_positions, joints, axis=0)
    node.spheres.add_frames(joints)
    node.lines.add_frames(joints[:, node.skeleton].reshape(len(joints), -1, 3))


def reset_motion_for_new_stream(state) -> None:
    state.nodes.clearer.clear_motion()
    state.stream_buffers.accum_joints = None
    state.stream_buffers.trimmed_frames = 0


def accumulate_joints(stream_buffers, new: np.ndarray) -> None:
    stream_buffers.accum_joints = (
        new
        if stream_buffers.accum_joints is None
        else np.concatenate([stream_buffers.accum_joints, new], axis=0)
    )


def add_or_extend_motion(nodes, new: np.ndarray) -> None:
    if nodes.motion is None:
        motion = create_skeleton_sequence_node(new)
        motion.name = "Motion (streaming)"
        nodes.show_motion(motion)
    else:
        append_skeleton_sequence_frames(nodes.motion, new)


def start_stream_view(state, host: ViewerHost) -> None:
    state.stream_buffers.started = True
    state.nodes.clearer.clear_body()
    state.nodes.clearer.clear_rest()
    state.nodes.follow_center = None
    host.scene.current_frame_id = 0
    host.toggle_animation(True)
    state.stream_buffers.need_reset_view = True


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
        self.clearer = SceneNodeClearer(scene, self)

    def show_rest(self, node) -> None:
        self.clearer.clear_rest()
        self.rest = node
        self.scene.add(node)

    def show_motion(self, node) -> None:
        self.clearer.clear_motion()
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
        self.camera = StreamCamera(host, state)

    def consume(self) -> None:
        chunks, body_chunks, remesh = self.state.stream_buffers.drain()

        if chunks:
            self.append_chunks(chunks)
        if body_chunks:
            self.append_body_chunks(body_chunks)
        if remesh is not None:
            self.apply_remesh(*remesh)
        if self.state.stream_buffers.need_reset_view:
            self.state.stream_buffers.need_reset_view = False
            node = self.state.nodes.focus
            if node is not None:
                self.camera.center_view(node)

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

    def append_chunks(self, chunks: list[np.ndarray]) -> None:
        new = np.concatenate(chunks, axis=0)
        if not self.state.stream_buffers.started:
            reset_motion_for_new_stream(self.state)
        accumulate_joints(self.state.stream_buffers, new)
        add_or_extend_motion(self.state.nodes, new)
        logger.info(
            f"[DISP] skeleton node added: accum={self.state.stream_buffers.accum_joints.shape} "
            f"scene_frames={self.host.scene.n_frames} started={self.state.stream_buffers.started}"
        )
        self.trim_to_window()
        if not self.state.stream_buffers.started:
            start_stream_view(self.state, self.host)

    def trim_to_window(self) -> None:
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

    def append_body_chunks(self, chunks: list[tuple[np.ndarray, np.ndarray]]) -> None:
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

    def apply_remesh(self, verts: np.ndarray, faces: np.ndarray, is_rest: bool) -> None:
        if is_rest:
            if self.state.nodes.has_generated_motion:
                return
            self.state.nodes.show_rest(
                Meshes(verts, faces, color=self.state.skin_color, name="Avatar (rest)")
            )
            return
        frame = self.host.scene.current_frame_id
        self.state.nodes.clearer.clear_all()
        self.state.nodes.show_body(
            Meshes(verts, faces, color=self.state.skin_color, name="Avatar (SMPL-X)")
        )
        self.state.stream_buffers.body_verts = verts
        self.host.scene.current_frame_id = min(frame, len(verts) - 1)
