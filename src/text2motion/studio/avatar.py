from __future__ import annotations

import threading
import traceback
from dataclasses import replace

import numpy as np
import torch

from text2motion.studio.contracts import SmplxFitResult, ViewerHost
from text2motion.studio.fitting import (
    build_smplx_body_model,
    smplx_mesh_from_fit_parameters,
    solve_smplx_pose_from_joints,
    upsample_pose_parameters,
)
from text2motion.studio.scene import create_skeleton_sequence_node


############################################################
#       SmplxAvatarController
############################################################
class SmplxAvatarController:
    def __init__(self, host: ViewerHost, state) -> None:
        self.host = host
        self.state = state

    @property
    def schedule(self):
        return self.state.config.schedule

    def fit_single_chunk(self, joints: np.ndarray, n_frames_so_far: int, factor: int) -> None:
        try:
            self.__fitted_chunk(joints, n_frames_so_far, factor)
        except Exception as exc:
            traceback.print_exc()
            self.state.status = (
                f"streaming... {n_frames_so_far} frames but SMPL-X fit failed for a chunk: "
                f"{type(exc).__name__}: {exc}"
            )

    def __body_model(self, frames: int, gender: str):
        model = self.state.smplx_fit.model_cache.get((frames, gender))
        if model is None:
            cfg = replace(self.state.config.fit, model_dir=self.state.model_dir, gender=gender)
            model = build_smplx_body_model(cfg, frames, self.state.device)
            self.state.smplx_fit.model_cache[(frames, gender)] = model
        return model

    def __display_rate_result(
        self, res: SmplxFitResult, factor: int, gender: str
    ) -> SmplxFitResult:
        if factor == 1:
            return res
        buffers = self.state.stream_buffers
        orient, pose, transl = upsample_pose_parameters(
            buffers.fit_anchor, res.global_orient, res.body_pose, res.transl, factor
        )
        buffers.fit_anchor = (res.global_orient[-1], res.body_pose[-1], res.transl[-1])
        model = self.__body_model(len(orient), gender)
        dev = self.state.device if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            out = model(
                global_orient=torch.tensor(orient, device=dev),
                body_pose=torch.tensor(pose, device=dev),
                transl=torch.tensor(transl, device=dev),
                betas=torch.tensor(res.betas, dtype=torch.float32, device=dev)
                .unsqueeze(0)
                .expand(len(orient), -1),
            )
        return replace(
            res,
            vertices=out.vertices.cpu().numpy(),
            global_orient=orient,
            body_pose=pose,
            transl=transl,
        )

    def __fitted_chunk(self, joints: np.ndarray, n_frames_so_far: int, factor: int) -> None:
        gender = self.state.gender
        frames = joints.shape[0]
        if self.state.device == "cuda":
            torch.cuda.empty_cache()
        model = self.__body_model(frames, gender)
        if self.state.stream_buffers.fit_is_stale():
            return
        res = solve_smplx_pose_from_joints(model, joints, self.state.device)
        self.state.stream_buffers.fit_progress = 1.0
        display = self.__display_rate_result(res, factor, gender)
        display_frames = len(display.global_orient)
        display_verts = self.__shaped_vertices(
            self.__body_model(display_frames, gender), display, display_frames
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

    def __shaped_vertices(self, model, res, t: int) -> np.ndarray:
        user = self.state.user_betas
        if not user.any():
            return res.vertices
        dev = self.state.device if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            out = model(
                global_orient=torch.tensor(res.global_orient, device=dev),
                body_pose=torch.tensor(res.body_pose, device=dev),
                transl=torch.tensor(res.transl, device=dev),
                betas=torch.tensor(res.betas + user, dtype=torch.float32, device=dev)
                .unsqueeze(0)
                .expand(t, -1),
            )
        return out.vertices.cpu().numpy()

    def on_avatar_changed(self) -> None:
        if not self.state.model_dir or not self.state.fit_body:
            return
        if self.state.generating:
            self.state.smplx_fit.after_gen = True
            self.state.smplx_fit.note = (
                "new chunks use the new avatar; full clip refreshes when done"
            )
            return
        self.schedule_remesh()

    def on_fit_body_toggled(self) -> None:
        if self.state.generating:
            self.state.smplx_fit.note = "body-fit choice applies to the next generation"
            return
        if not self.state.fit_body:
            self.__show_skeleton_only()
            return
        if self.state.stream_buffers.accum_joints is None or self.state.smplx_fit.orient:
            self.schedule_remesh()
            return
        self.__start_retrofit()

    def __show_skeleton_only(self) -> None:
        self.state.nodes.clear_body()
        self.state.nodes.clear_rest()
        joints = self.state.stream_buffers.accum_joints
        if joints is None or self.state.nodes.motion is not None:
            return
        node = create_skeleton_sequence_node(joints)
        node.name = "Motion (streaming)"
        self.state.nodes.motion = node
        self.host.scene.add(node)

    def __start_retrofit(self) -> None:
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
            target=self.__fit_existing_motion,
            args=(source[-frames - 1 :].copy(), frames),
            daemon=True,
        ).start()

    def __anchor_pose(self, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self.__body_model(len(anchor), self.state.gender)
        res = solve_smplx_pose_from_joints(model, anchor, self.state.device)
        return res.global_orient[-1], res.body_pose[-1], res.transl[-1]

    def __fit_existing_motion(self, source: np.ndarray, frames: int) -> None:
        factor = self.state.upsample
        window = max(1, self.schedule.retrofit_window // factor)
        anchor, motion = source[:-frames], source[-frames:]
        n_frames = 0
        try:
            if len(anchor):
                self.state.stream_buffers.fit_anchor = self.__anchor_pose(anchor)
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

    def schedule_remesh(self) -> None:
        self.state.smplx_fit.version += 1
        if self.state.smplx_fit.running:
            return
        self.state.smplx_fit.running = True
        threading.Thread(target=self.__rebuild_avatar_mesh, daemon=True).start()

    def __rebuild_avatar_mesh(self) -> None:
        fit = self.state.smplx_fit
        try:
            while fit.done_version != fit.version:
                version = fit.version
                self.__update_avatar_mesh()
                fit.done_version = version
        finally:
            fit.running = False
        if fit.done_version != fit.version:
            self.schedule_remesh()

    def __update_avatar_mesh(self) -> None:
        gender = self.state.gender
        user_betas = self.state.user_betas.copy()
        try:
            self.state.smplx_fit.note = f"updating avatar ({gender})..."
            pending = self.__avatar_mesh(gender, user_betas)
            with self.state.stream_buffers.lock:
                self.state.stream_buffers.pending_remesh = pending
            self.state.smplx_fit.note = ""
        except Exception as exc:
            traceback.print_exc()
            self.state.smplx_fit.note = f"avatar update failed: {type(exc).__name__}: {exc}"

    def __fit_parameters(self):
        fit = self.state.smplx_fit
        with self.state.stream_buffers.lock:
            if not fit.orient:
                return None
            if not fit.pose or not fit.transl:
                raise ValueError("smplx fit lacks pose or translation")
            betas = None if fit.betas is None else fit.betas.copy()
            orient = np.concatenate(fit.orient, 0)
            return orient, np.concatenate(fit.pose, 0), np.concatenate(fit.transl, 0), betas

    def __avatar_mesh(
        self, gender: str, user_betas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        cfg = replace(self.state.config.fit, model_dir=self.state.model_dir, gender=gender)
        parameters = self.__fit_parameters()
        if parameters is None:
            zero = np.zeros((1, 3), dtype=np.float32)
            rest_pose = np.zeros((1, 63), dtype=np.float32)
            vertices, faces = smplx_mesh_from_fit_parameters(
                cfg, zero, rest_pose, zero, user_betas, gender, "cpu"
            )
            return vertices, faces, True
        orient, pose, transl, fit_betas = parameters
        betas = user_betas if fit_betas is None else fit_betas + user_betas
        vertices, faces = smplx_mesh_from_fit_parameters(
            cfg, orient, pose, transl, betas.astype(np.float32), gender, self.state.device
        )
        return vertices, faces, False
