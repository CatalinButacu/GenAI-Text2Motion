from __future__ import annotations

import threading
import traceback

import numpy as np
import torch

from text2motion.studio.avatar_body import SmplxBodyModels
from text2motion.studio.avatar_mesh import AvatarMeshRebuilder
from text2motion.studio.contracts import ViewerHost
from text2motion.studio.pose_parameters import solve_smplx_pose_from_joints
from text2motion.studio.scene import create_skeleton_sequence_node


############################################################
#       SmplxAvatarController
############################################################
class SmplxAvatarController:
    def __init__(self, host: ViewerHost, state) -> None:
        self.host = host
        self.state = state
        self.mesh = AvatarMeshRebuilder(state)
        self.body = SmplxBodyModels(state)

    @property
    def schedule(self):
        return self.state.config.schedule

    def fit_single_chunk(self, joints: np.ndarray, n_frames_so_far: int, factor: int) -> None:
        try:
            self.fitted_chunk(joints, n_frames_so_far, factor)
        except Exception as exc:
            traceback.print_exc()
            self.state.status = (
                f"streaming... {n_frames_so_far} frames but SMPL-X fit failed for a chunk: "
                f"{type(exc).__name__}: {exc}"
            )

    def fitted_chunk(self, joints: np.ndarray, n_frames_so_far: int, factor: int) -> None:
        gender = self.state.gender
        frames = joints.shape[0]
        if self.state.device == "cuda":
            torch.cuda.empty_cache()
        model = self.body.body_model(frames, gender)
        if self.state.stream_buffers.fit_is_stale():
            return
        res = solve_smplx_pose_from_joints(model, joints, self.state.device)
        self.state.stream_buffers.fit_progress = 1.0
        display = self.body.display_rate_result(res, factor, gender)
        display_frames = len(display.global_orient)
        display_verts = self.body.shaped_vertices(
            self.body.body_model(display_frames, gender), display, display_frames
        )
        appended = self.state.stream_buffers.append_fitted_chunk(
            self.state.smplx_fit, display_verts, display.faces, display
        )
        if not appended:
            return
        duration_so_far = n_frames_so_far / self.host.playback_fps
        self.state.status = (
            f"{'streaming' if self.state.generating else 'fitting body'}... "
            f"{n_frames_so_far} frames (~{duration_so_far:.1f}s), "
            f"body fit {res.joint_err_cm:.1f} cm"
        )

    def on_avatar_changed(self) -> None:
        if not self.state.model_dir or not self.state.fit_body:
            return
        if self.state.generating:
            self.state.smplx_fit.after_gen = True
            self.state.smplx_fit.note = (
                "new chunks use the new avatar; full clip refreshes when done"
            )
            return
        self.mesh.schedule_remesh()

    def on_fit_body_toggled(self) -> None:
        if self.state.generating:
            self.state.smplx_fit.note = "body-fit choice applies to the next generation"
            return
        if not self.state.fit_body:
            self.show_skeleton_only()
            return
        if self.state.stream_buffers.accum_joints is None or self.state.smplx_fit.orient:
            self.mesh.schedule_remesh()
            return
        self.start_retrofit()

    def show_skeleton_only(self) -> None:
        self.state.nodes.clear_body()
        self.state.nodes.clear_rest()
        joints = self.state.stream_buffers.accum_joints
        if joints is None or self.state.nodes.motion is not None:
            return
        node = create_skeleton_sequence_node(joints)
        node.name = "Motion (streaming)"
        self.state.nodes.motion = node
        self.host.scene.add(node)

    def start_retrofit(self) -> None:
        joints = self.state.stream_buffers.accum_joints
        source = self.state.stream_buffers.source_joints
        if joints is None or source is None or self.state.generating:
            return
        factor = self.state.upsample
        if len(joints) % factor:
            raise ValueError(
                f"{len(joints)} display frames are not a multiple of the upsample factor {factor}"
            )
        frames = len(joints) // factor
        self.state.generating = True
        self.state.stream_buffers.cancel = False
        self.state.stream_buffers.current_fit_body = True
        self.state.stream_buffers.gen_progress = 1.0
        with self.state.stream_buffers.lock:
            self.state.stream_buffers.take += 1
            self.state.stream_buffers.fit_take = self.state.stream_buffers.take
        self.state.stream_buffers.fitting = True
        self.state.status = "fitting SMPL-X to the generated motion..."
        self.state.stream_buffers.reset_fit_track(self.state.smplx_fit)
        threading.Thread(
            target=self.fit_existing_motion,
            args=(source[-frames - 1 :].copy(), frames),
            daemon=True,
        ).start()

    def anchor_pose(self, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self.body.body_model(len(anchor), self.state.gender)
        res = solve_smplx_pose_from_joints(model, anchor, self.state.device)
        return res.global_orient[-1], res.body_pose[-1], res.transl[-1]

    def fit_existing_motion(self, source: np.ndarray, frames: int) -> None:
        factor = self.state.upsample
        window = max(1, self.schedule.retrofit_window // factor)
        anchor, motion = source[:-frames], source[-frames:]
        n_frames = 0
        try:
            if len(anchor):
                self.state.stream_buffers.fit_anchor = self.anchor_pose(anchor)
            for start in range(0, len(motion), window):
                if self.state.stream_buffers.cancel:
                    break
                piece = motion[start : start + window]
                n_frames += len(piece) * factor
                self.fit_single_chunk(piece, n_frames, factor)
        finally:
            verb = "stopped" if self.state.stream_buffers.cancel else "done"
            self.state.status = f"body fit {verb} -- {n_frames} frames"
            self.state.stream_buffers.fit_progress = None
            self.state.stream_buffers.fitting = False
            if not self.host.run_animations:
                self.host.toggle_animation(True)
            self.state.generating = False
