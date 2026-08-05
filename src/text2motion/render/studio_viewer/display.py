from __future__ import annotations

import numpy as np
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import ViewerCamera

from text2motion.render.studio import build_skeleton_seq


class Display:
    def __init__(self, host, state) -> None:
        self.host = host
        self.state = state

    def _consume(self) -> None:
        chunks, body_chunks, remesh = self.state.buf.drain()

        if chunks:
            self._append_chunks(chunks)
        if body_chunks:
            self._append_body_chunks(body_chunks)
        if self.state.buf.drop_skeleton and not body_chunks:
            self.state.buf.drop_skeleton = False
            if self.state.nodes.body is not None:
                self.state.nodes.clear_motion()
                self.state.nodes.body.enabled = True
        if remesh is not None:
            self._apply_remesh(*remesh)
        if self.state.buf.need_reset_view:
            self.state.buf.need_reset_view = False
            node = self.state.nodes.focus
            if node is not None:
                self._center_view(node)

    def _sync_display_nodes(self) -> None:
        if self.state.nodes.motion is None or self.state.nodes.body is None:
            return
        body_frames = 0 if self.state.buf.body_verts is None else len(self.state.buf.body_verts)
        on_body = self.host.scene.current_frame_id < body_frames
        self.state.nodes.body.enabled = on_body
        self.state.nodes.motion.enabled = not on_body

    def _center_view(self, node) -> None:
        bounds = node.current_bounds
        if not np.isfinite(bounds).all():  # degenerate/empty geometry: centring would divide by nan
            print(f"[warn] skipped centring on {node.name}: non-finite bounds")
            return
        self.host.center_view_on_node(node)

    def _append_chunks(self, chunks: list[np.ndarray]) -> None:
        new = np.concatenate(chunks, axis=0)
        self.state.buf.accum_joints = (
            new
            if self.state.buf.accum_joints is None
            else np.concatenate([self.state.buf.accum_joints, new], axis=0)
        )
        motion = build_skeleton_seq(self.state.buf.accum_joints)
        motion.name = "Motion (streaming)"
        self.state.nodes.show_motion(motion)
        print(
            f"[DISP] skeleton node added: accum={self.state.buf.accum_joints.shape} "
            f"scene_frames={self.host.scene.n_frames} started={self.state.buf.started}",
            flush=True,
        )
        if not self.state.buf.started:
            self.state.buf.started = True
            self.state.nodes.clear_body()
            self.state.nodes.clear_rest()
            self.state.nodes.follow_center = None
            self.host.scene.current_frame_id = 0
            self.host.toggle_animation(True)
            self.state.buf.need_reset_view = True

    def _append_body_chunks(self, chunks: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for verts, faces in chunks:
            if self.state.nodes.body is None:
                self.state.nodes.show_body(
                    Meshes(verts, faces, color=self.state.skin_color, name="Avatar (SMPL-X)")
                )
                self.state.buf.body_verts = verts
            else:
                self.state.nodes.body.add_frames(verts)
                self.state.buf.body_verts = self.state.nodes.body.vertices

    def _apply_remesh(self, verts: np.ndarray, faces: np.ndarray, is_rest: bool) -> None:
        if is_rest:
            if self.state.nodes.has_motion_or_body:
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
        self.state.buf.body_verts = verts
        self.host.scene.current_frame_id = min(frame, len(verts) - 1)

    def _current_center(self) -> np.ndarray | None:
        frame = self.host.scene.current_frame_id
        body_frames = 0 if self.state.buf.body_verts is None else len(self.state.buf.body_verts)
        if self.state.nodes.body is not None and (
            self.state.nodes.motion is None or frame < body_frames
        ):
            arr = self.state.buf.body_verts
        elif self.state.nodes.motion is not None and self.state.buf.accum_joints is not None:
            arr = self.state.buf.accum_joints
        else:
            return None
        return arr[min(frame, len(arr) - 1)].mean(axis=0)

    def _update_follow_cam(self) -> None:
        if not self.state.follow_cam:
            return
        camera = self.host.scene.camera
        if not isinstance(camera, ViewerCamera):
            return
        center = self._current_center()
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
