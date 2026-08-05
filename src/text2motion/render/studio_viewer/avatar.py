from __future__ import annotations

import threading
import traceback

import numpy as np
import torch

from text2motion.render.joints2smpl import (
    FitConfig,
    build_smplx_model,
    fit_smplx_to_joints,
    mesh_from_params,
)
from text2motion.render.studio import build_skeleton_seq


class Avatar:
    def __init__(self, host, state) -> None:
        self.host = host
        self.state = state

    def _fit_chunk(
        self,
        joints: np.ndarray,
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        n_frames_so_far: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        is_first = warm_start is None
        gender = self.state.gender
        cfg = FitConfig(
            model_dir=self.state.model_dir,
            gender=gender,
            stage1_iters=60 if is_first else 0,
            stage2_iters=150 if is_first else 25,
        )
        duration_so_far = n_frames_so_far / self.host.playback_fps
        t = joints.shape[0]
        try:
            if self.state.device == "cuda":
                torch.cuda.empty_cache()

            model = self.state.fit.model_cache.get((t, gender))
            if model is None:
                model = build_smplx_model(cfg, t, self.state.device)
                self.state.fit.model_cache[(t, gender)] = model

            with torch.enable_grad():
                res = fit_smplx_to_joints(
                    joints,
                    cfg,
                    device=self.state.device,
                    warm_start=warm_start,
                    model=model,
                    on_progress=self._on_fit_progress,
                )

            display_verts = self._shaped_vertices(model, res, t)
            self.state.buf.append_fitted_chunk(self.state.fit, display_verts, res.faces, res)

            self.state.status = (
                f"streaming... {n_frames_so_far} frames (~{duration_so_far:.1f}s), "
                f"body fit {res.joint_err_cm:.1f} cm"
            )
            return (res.global_orient[-1], res.body_pose[-1], res.betas)
        except Exception as exc:
            traceback.print_exc()
            self.state.status = (
                f"streaming... {n_frames_so_far} frames but SMPL-X fit failed for a chunk: "
                f"{type(exc).__name__}: {exc}"
            )
            return warm_start

    def _shaped_vertices(self, model, res, t: int) -> np.ndarray:
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

    def _on_fit_progress(self, done: int, total: int) -> None:
        self.state.buf.fit_progress = done / max(1, total)

    def _on_avatar_changed(self) -> None:
        if not self.state.model_dir or not self.state.fit_body:
            return
        if self.state.generating:
            self.state.fit.after_gen = True
            self.state.fit.note = "new chunks use the new avatar; full clip refreshes when done"
            return
        self._schedule_remesh()

    def _on_fit_body_toggled(self) -> None:
        if self.state.generating:
            self.state.fit.note = "body-fit choice applies to the next generation"
            return
        if not self.state.fit_body:
            self.state.nodes.clear_body()
            self.state.nodes.clear_rest()
            if self.state.buf.accum_joints is not None and self.state.nodes.motion is None:
                self.state.nodes.motion = build_skeleton_seq(self.state.buf.accum_joints)
                self.state.nodes.motion.name = "Motion (streaming)"
                self.host.scene.add(self.state.nodes.motion)
            return
        if self.state.buf.accum_joints is None or self.state.fit.orient:
            self._schedule_remesh()
        else:
            self._start_retrofit()

    def _start_retrofit(self) -> None:
        joints = self.state.buf.accum_joints
        if joints is None or self.state.generating:
            return
        self.state.generating = True
        self.state.buf.cancel = False
        self.state.buf.current_fit_body = True
        self.state.buf.gen_progress = 1.0
        self.state.buf.drop_skeleton = False
        self.state.status = "fitting SMPL-X to the generated motion..."
        self.state.buf.reset_fit_track(self.state.fit)
        threading.Thread(target=self._retrofit_worker, args=(joints.copy(),), daemon=True).start()

    def _retrofit_worker(self, joints: np.ndarray) -> None:
        warm_start = None
        window = 16
        n_frames = 0
        try:
            for start in range(0, len(joints), window):
                if self.state.buf.cancel:
                    break
                piece = joints[start : start + window]
                n_frames += len(piece)
                warm_start = self._fit_chunk(piece, warm_start, n_frames)
        finally:
            verb = "stopped" if self.state.buf.cancel else "done"
            self.state.status = f"body fit {verb} -- {n_frames} frames"
            self.state.buf.fit_progress = None
            if not self.state.buf.cancel:
                self.state.buf.drop_skeleton = True
            if not self.host.run_animations:
                self.host.toggle_animation(True)
            self.state.generating = False

    def _schedule_remesh(self) -> None:
        self.state.fit.version += 1
        if self.state.fit.running:
            return
        self.state.fit.running = True
        threading.Thread(target=self._remesh_worker, daemon=True).start()

    def _remesh_worker(self) -> None:
        try:
            while self.state.fit.done_version != self.state.fit.version:
                version = self.state.fit.version
                gender = self.state.gender
                user = self.state.user_betas.copy()
                with self.state.buf.lock:
                    orient = (
                        np.concatenate(self.state.fit.orient, 0) if self.state.fit.orient else None
                    )
                    pose = np.concatenate(self.state.fit.pose, 0) if self.state.fit.pose else None
                    transl = (
                        np.concatenate(self.state.fit.transl, 0) if self.state.fit.transl else None
                    )
                    fit_betas = (
                        None if self.state.fit.betas is None else self.state.fit.betas.copy()
                    )
                cfg = FitConfig(model_dir=self.state.model_dir, gender=gender)
                try:
                    self.state.fit.note = f"updating avatar ({gender})..."
                    if orient is None:
                        zero = np.zeros((1, 3), dtype=np.float32)
                        verts, faces = mesh_from_params(
                            cfg,
                            zero,
                            np.zeros((1, 63), dtype=np.float32),
                            zero,
                            user,
                            gender,
                            "cpu",
                        )
                        is_rest = True
                    else:
                        betas = user if fit_betas is None else fit_betas + user
                        verts, faces = mesh_from_params(
                            cfg,
                            orient,
                            pose,
                            transl,
                            betas.astype(np.float32),
                            gender,
                            self.state.device,
                        )
                        is_rest = False
                    with self.state.buf.lock:
                        self.state.buf.pending_remesh = (verts, faces, is_rest)
                    self.state.fit.note = ""
                except Exception as exc:
                    traceback.print_exc()
                    self.state.fit.note = f"avatar update failed: {type(exc).__name__}: {exc}"
                self.state.fit.done_version = version
        finally:
            self.state.fit.running = False
        if self.state.fit.done_version != self.state.fit.version:
            self._schedule_remesh()
