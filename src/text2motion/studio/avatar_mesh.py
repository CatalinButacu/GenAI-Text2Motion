from __future__ import annotations

import threading
import traceback
from dataclasses import replace

import numpy as np

from text2motion.studio.fitting import smplx_mesh_from_fit_parameters


############################################################
#       AvatarMeshRebuilder
############################################################
class AvatarMeshRebuilder:
    def __init__(self, state) -> None:
        self.state = state

    def schedule_remesh(self) -> None:
        self.state.smplx_fit.version += 1
        if self.state.smplx_fit.running:
            return
        self.state.smplx_fit.running = True
        threading.Thread(target=self.rebuild_avatar_mesh, daemon=True).start()

    def rebuild_avatar_mesh(self) -> None:
        fit = self.state.smplx_fit
        try:
            while fit.done_version != fit.version:
                version = fit.version
                self.update_avatar_mesh()
                fit.done_version = version
        finally:
            fit.running = False
        if fit.done_version != fit.version:
            self.schedule_remesh()

    def update_avatar_mesh(self) -> None:
        gender = self.state.gender
        user_betas = self.state.user_betas.copy()
        try:
            self.state.smplx_fit.note = f"updating avatar ({gender})..."
            pending = self.avatar_mesh(gender, user_betas)
            with self.state.stream_buffers.lock:
                self.state.stream_buffers.pending_remesh = pending
            self.state.smplx_fit.note = ""
        except Exception as exc:
            traceback.print_exc()
            self.state.smplx_fit.note = f"avatar update failed: {type(exc).__name__}: {exc}"

    def fit_parameters(self):
        fit = self.state.smplx_fit
        with self.state.stream_buffers.lock:
            if not fit.orient:
                return None
            if not fit.pose or not fit.transl:
                raise ValueError("smplx fit lacks pose or translation")
            betas = None if fit.betas is None else fit.betas.copy()
            orient = np.concatenate(fit.orient, 0)
            return orient, np.concatenate(fit.pose, 0), np.concatenate(fit.transl, 0), betas

    def avatar_mesh(
        self, gender: str, user_betas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        cfg = replace(self.state.config.fit, model_dir=self.state.model_dir, gender=gender)
        parameters = self.fit_parameters()
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
