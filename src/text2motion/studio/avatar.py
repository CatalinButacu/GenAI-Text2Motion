from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import smplx
import torch

from text2motion.motion.model import SMPLX_MODEL_TYPE
from text2motion.motion.representation import JOINTS, kinematic_bones
from text2motion.studio.config import FitConfig
from text2motion.studio.contracts import ViewerHost


def build_skeleton_seq(joints: np.ndarray):
    from aitviewer.renderables.skeletons import Skeletons

    return Skeletons(joint_positions=joints, joint_connections=kinematic_bones())


def append_skeleton_frames(node, joints: np.ndarray) -> None:
    node.joint_positions = np.append(node.joint_positions, joints, axis=0)
    node.spheres.add_frames(joints)
    node.lines.add_frames(joints[:, node.skeleton].reshape(len(joints), -1, 3))


@dataclass
class FitResult:
    vertices: np.ndarray
    faces: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    joint_err_cm: float


def build_smplx_model(cfg: FitConfig, t: int, device: str):
    return smplx.create(
        cfg.model_dir,
        model_type=SMPLX_MODEL_TYPE,
        gender=cfg.gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=cfg.num_betas,
        ext="npz",
        batch_size=t,
    ).to(device)


def _yaw_init(
    model,
    target: torch.Tensor,
    transl: torch.Tensor,
    device: str,
    candidates_deg: tuple[float, ...],
) -> torch.Tensor:
    t = target.shape[0]
    best_aa, best_err = None, float("inf")
    with torch.no_grad():
        for yaw in np.deg2rad(candidates_deg):
            aa = torch.zeros(t, 3, device=device)
            aa[:, 1] = yaw
            out = model(
                global_orient=aa,
                body_pose=torch.zeros(t, 63, device=device),
                transl=transl,
                betas=torch.zeros(t, model.num_betas, device=device),
            )
            err = (out.joints[:, :JOINTS] - target).pow(2).sum(-1).mean().item()
            if err < best_err:
                best_err, best_aa = err, aa.clone()
    return best_aa


def rest_pose_body(cfg: FitConfig, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model = build_smplx_model(cfg, 1, device)
    with torch.no_grad():
        out = model(
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            transl=torch.zeros(1, 3, device=device),
            betas=torch.zeros(1, cfg.num_betas, device=device),
        )
    return out.vertices[0].cpu().numpy(), model.faces.astype(np.int64)


def mesh_from_params(
    cfg: FitConfig,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray,
    gender: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    t = global_orient.shape[0]
    model = build_smplx_model(replace(cfg, gender=gender), t, device)
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(global_orient, device=device),
            body_pose=torch.tensor(body_pose, device=device),
            transl=torch.tensor(transl, device=device),
            betas=torch.tensor(betas, dtype=torch.float32, device=device).expand(t, -1),
        )
    return out.vertices.cpu().numpy(), model.faces.astype(np.int64)


def fit_smplx_to_joints(
    target_joints: np.ndarray,
    cfg: FitConfig,
    device: str = "cuda",
    verbose: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    model=None,
) -> FitResult:
    dev = device if torch.cuda.is_available() else "cpu"
    t = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=dev)
    if model is None:
        model = build_smplx_model(cfg, t, dev)
    total_iters = max(1, cfg.stage1_iters + cfg.stage2_iters)

    transl = target[:, 0].clone().requires_grad_(True)
    if warm_start is not None:
        last_orient, last_pose, last_betas = warm_start
        global_orient = (
            torch.tensor(last_orient, dtype=torch.float32, device=dev)
            .expand(t, -1)
            .clone()
            .requires_grad_(True)
        )
        body_pose = (
            torch.tensor(last_pose, dtype=torch.float32, device=dev)
            .expand(t, -1)
            .clone()
            .requires_grad_(True)
        )
        betas = torch.tensor(last_betas, dtype=torch.float32, device=dev).unsqueeze(0)
    else:
        global_orient = _yaw_init(
            model, target, target[:, 0].detach(), dev, cfg.yaw_candidates_deg
        ).requires_grad_(True)
        body_pose = torch.zeros(t, 63, device=dev, requires_grad=True)
        betas = torch.zeros(1, cfg.num_betas, device=dev, requires_grad=True)

    def forward():
        return model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas.expand(t, -1),
        ).joints[:, :JOINTS]

    if cfg.stage1_iters > 0:
        stage1 = torch.optim.Adam([global_orient, transl], lr=cfg.lr)
        for it in range(cfg.stage1_iters):
            stage1.zero_grad()
            ((forward() - target) ** 2).sum(-1).mean().backward()
            stage1.step()
            if on_progress is not None and (
                it % cfg.progress_every == 0 or it == cfg.stage1_iters - 1
            ):
                on_progress(it, total_iters)

    stage2_params = [global_orient, body_pose, transl] + ([] if warm_start is not None else [betas])
    stage2 = torch.optim.Adam(stage2_params, lr=cfg.lr)
    for it in range(cfg.stage2_iters):
        stage2.zero_grad()
        joints = forward()
        data = ((joints - target) ** 2).sum(-1).mean()
        smooth = (body_pose[1:] - body_pose[:-1]).pow(2).mean() + (
            global_orient[1:] - global_orient[:-1]
        ).pow(2).mean()
        loss = data + cfg.w_smooth * smooth + cfg.w_reg * body_pose.pow(2).mean()
        loss.backward()
        stage2.step()
        if on_progress is not None and (it % cfg.progress_every == 0 or it == cfg.stage2_iters - 1):
            on_progress(cfg.stage1_iters + it, total_iters)
        if verbose and (it % cfg.verbose_every == 0 or it == cfg.stage2_iters - 1):
            err = (joints.detach() - target).norm(dim=-1).mean().item() * 100
            print(f"  fit it {it:3d}  joint err {err:.1f} cm")

    with torch.no_grad():
        out = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas.expand(t, -1),
        )
        err_cm = (out.joints[:, :JOINTS] - target).norm(dim=-1).mean().item() * 100
    return FitResult(
        vertices=out.vertices.cpu().numpy(),
        faces=model.faces.astype(np.int64),
        global_orient=global_orient.detach().cpu().numpy(),
        body_pose=body_pose.detach().cpu().numpy(),
        transl=transl.detach().cpu().numpy(),
        betas=betas.detach().cpu().numpy()[0],
        joint_err_cm=err_cm,
    )


class Avatar:
    def __init__(self, host: ViewerHost, state) -> None:
        self.host = host
        self.state = state

    @property
    def schedule(self):
        return self.state.config.schedule

    def _chunk_config(self, gender: str, is_first: bool) -> FitConfig:
        schedule = self.schedule
        return replace(
            self.state.config.fit,
            model_dir=self.state.model_dir,
            gender=gender,
            stage1_iters=(
                schedule.first_stage1_iters if is_first else schedule.continuation_stage1_iters
            ),
            stage2_iters=(
                schedule.first_stage2_iters if is_first else schedule.continuation_stage2_iters
            ),
        )

    def _fit_chunk(
        self,
        joints: np.ndarray,
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        n_frames_so_far: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        gender = self.state.gender
        cfg = self._chunk_config(gender, is_first=warm_start is None)
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
        window = self.schedule.retrofit_window
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
                cfg = replace(self.state.config.fit, model_dir=self.state.model_dir, gender=gender)
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
