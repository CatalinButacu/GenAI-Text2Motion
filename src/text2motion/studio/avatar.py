from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import smplx
import torch

from text2motion.motion.model import SMPLX_MODEL_TYPE
from text2motion.motion.representation import JOINTS, kinematic_bones
from text2motion.studio.config import SmplxFitConfig
from text2motion.studio.contracts import ViewerHost

logger = logging.getLogger(__name__)


class FitCancelled(Exception):
    pass


def create_skeleton_sequence_node(joints: np.ndarray):
    from aitviewer.renderables.skeletons import Skeletons

    return Skeletons(joint_positions=joints, joint_connections=kinematic_bones())


def append_skeleton_sequence_frames(node, joints: np.ndarray) -> None:
    node.joint_positions = np.append(node.joint_positions, joints, axis=0)
    node.spheres.add_frames(joints)
    node.lines.add_frames(joints[:, node.skeleton].reshape(len(joints), -1, 3))


@dataclass
class SmplxFitResult:
    vertices: np.ndarray
    faces: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    joint_err_cm: float


def build_smplx_body_model(cfg: SmplxFitConfig, t: int, device: str):
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


def __yaw_init(
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
    if best_aa is None:
        raise ValueError("yaw initialisation found no candidate; candidates_deg must be non-empty")
    return best_aa


def create_smplx_rest_pose_mesh(
    cfg: SmplxFitConfig, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    model = build_smplx_body_model(cfg, 1, device)
    with torch.no_grad():
        out = model(
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            transl=torch.zeros(1, 3, device=device),
            betas=torch.zeros(1, cfg.num_betas, device=device),
        )
    return out.vertices[0].cpu().numpy(), model.faces.astype(np.int64)


def smplx_mesh_from_fit_parameters(
    cfg: SmplxFitConfig,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray,
    gender: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    t = global_orient.shape[0]
    model = build_smplx_body_model(replace(cfg, gender=gender), t, device)
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
    cfg: SmplxFitConfig,
    device: str = "cuda",
    verbose: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    model=None,
) -> SmplxFitResult:
    dev = device if torch.cuda.is_available() else "cpu"
    t = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=dev)
    if model is None:
        model = build_smplx_body_model(cfg, t, dev)
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
        global_orient = __yaw_init(
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
            logger.info(f"  fit it {it:3d}  joint err {err:.1f} cm")

    with torch.no_grad():
        out = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas.expand(t, -1),
        )
        err_cm = (out.joints[:, :JOINTS] - target).norm(dim=-1).mean().item() * 100
    return SmplxFitResult(
        vertices=out.vertices.cpu().numpy(),
        faces=model.faces.astype(np.int64),
        global_orient=global_orient.detach().cpu().numpy(),
        body_pose=body_pose.detach().cpu().numpy(),
        transl=transl.detach().cpu().numpy(),
        betas=betas.detach().cpu().numpy()[0],
        joint_err_cm=err_cm,
    )


def __kabsch_rotations(
    rest_vectors: torch.Tensor, target_vectors: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    covariance = torch.einsum("tk,tki,tkj->tij", weights, rest_vectors, target_vectors)
    u, _, vh = torch.linalg.svd(covariance)
    v = vh.transpose(-1, -2)
    reflection = torch.det(v @ u.transpose(-1, -2))
    signs = torch.stack([torch.ones_like(reflection), torch.ones_like(reflection), reflection], -1)
    return v @ torch.diag_embed(signs) @ u.transpose(-1, -2)


def rotation_matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = torch.stack(
        [
            torch.stack(
                [
                    1 + trace,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                    m[..., 0, 1] + m[..., 1, 0],
                    m[..., 0, 2] + m[..., 2, 0],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 0, 1] + m[..., 1, 0],
                    1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                    m[..., 1, 2] + m[..., 2, 1],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 1, 2] + m[..., 2, 1],
                    1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
                ],
                -1,
            ),
        ],
        -2,
    )
    diagonal = torch.stack(
        [
            1 + trace,
            1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ],
        -1,
    )
    best = diagonal.argmax(-1)
    quaternion = torch.gather(
        candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)
    ).squeeze(-2)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    sine = quaternion[..., 1:].norm(dim=-1)
    angle = 2 * torch.atan2(sine, quaternion[..., 0])
    axis = quaternion[..., 1:] / sine.clamp_min(1e-8).unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


SHORT_BONE_PARENT_PULL = {12: 1.0}


def solve_smplx_pose_from_joints(
    model, target_joints: np.ndarray, device: str, parent_pull: float = 0.05
) -> SmplxFitResult:
    frames = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)
    zeros_betas = torch.zeros(frames, model.num_betas, device=device)
    with torch.no_grad():
        rest = model(
            global_orient=torch.zeros(frames, 3, device=device),
            body_pose=torch.zeros(frames, 63, device=device),
            transl=torch.zeros(frames, 3, device=device),
            betas=zeros_betas,
        ).joints[0, :JOINTS]
        parents = model.parents[:JOINTS].tolist()
        children = {
            joint: [c for c in range(JOINTS) if parents[c] == joint] for joint in range(JOINTS)
        }
        identity = torch.eye(3, device=device).expand(frames, 3, 3)
        basis = torch.eye(3, device=device)
        global_rotation: list[torch.Tensor] = []
        for joint in range(JOINTS):
            parent_rotation = identity if joint == 0 else global_rotation[parents[joint]]
            rest_pairs, target_pairs, weight_pairs = [], [], []
            for child in children[joint]:
                for depth, node in enumerate([child] + children[child]):
                    rest_vector = rest[node] - rest[joint]
                    target_vector = target[:, node] - target[:, joint]
                    rest_pairs.append((rest_vector / rest_vector.norm()).expand(frames, 3))
                    target_pairs.append(
                        target_vector / target_vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    )
                    weight_pairs.append(
                        torch.full((frames,), 1.0 if depth == 0 else 0.5, device=device)
                    )
            if not rest_pairs:
                global_rotation.append(parent_rotation)
                continue
            for axis in range(3):
                rest_pairs.append(basis[axis].expand(frames, 3))
                target_pairs.append(parent_rotation @ basis[axis])
                weight_pairs.append(
                    torch.full(
                        (frames,), SHORT_BONE_PARENT_PULL.get(joint, parent_pull), device=device
                    )
                )
            global_rotation.append(
                __kabsch_rotations(
                    torch.stack(rest_pairs, 1),
                    torch.stack(target_pairs, 1),
                    torch.stack(weight_pairs, 1),
                )
            )
        local_rotation = [global_rotation[0]] + [
            global_rotation[parents[joint]].transpose(-1, -2) @ global_rotation[joint]
            for joint in range(1, JOINTS)
        ]
        axis_angle = rotation_matrix_to_axis_angle(torch.stack(local_rotation, 1))
        global_orient = axis_angle[:, 0]
        body_pose = axis_angle[:, 1:].reshape(frames, 63)
        transl = target[:, 0] - rest[0]
        out = model(
            global_orient=global_orient, body_pose=body_pose, transl=transl, betas=zeros_betas
        )
        err_cm = (out.joints[:, :JOINTS] - target).norm(dim=-1).mean().item() * 100
    return SmplxFitResult(
        vertices=out.vertices.cpu().numpy(),
        faces=model.faces.astype(np.int64),
        global_orient=global_orient.cpu().numpy(),
        body_pose=body_pose.cpu().numpy(),
        transl=transl.cpu().numpy(),
        betas=np.zeros(model.num_betas, dtype=np.float32),
        joint_err_cm=err_cm,
    )


class SmplxAvatarController:
    def __init__(self, host: ViewerHost, state) -> None:
        self.host = host
        self.state = state

    @property
    def schedule(self):
        return self.state.config.schedule

    def fit_single_chunk(
        self,
        joints: np.ndarray,
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        n_frames_so_far: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        gender = self.state.gender
        cfg = replace(self.state.config.fit, model_dir=self.state.model_dir, gender=gender)
        duration_so_far = n_frames_so_far / self.host.playback_fps
        t = joints.shape[0]
        try:
            if self.state.device == "cuda":
                torch.cuda.empty_cache()

            model = self.state.smplx_fit.model_cache.get((t, gender))
            if model is None:
                model = build_smplx_body_model(cfg, t, self.state.device)
                self.state.smplx_fit.model_cache[(t, gender)] = model

            if self.state.stream_buffers.fit_is_stale():
                raise FitCancelled
            res = solve_smplx_pose_from_joints(model, joints, self.state.device)
            self.state.stream_buffers.fit_progress = 1.0

            display_verts = self.__shaped_vertices(model, res, t)
            appended = self.state.stream_buffers.append_fitted_chunk(
                self.state.smplx_fit, display_verts, res.faces, res
            )
            if not appended:
                raise FitCancelled

            self.state.status = (
                f"{'streaming' if self.state.generating else 'fitting body'}... "
                f"{n_frames_so_far} frames (~{duration_so_far:.1f}s), "
                f"body fit {res.joint_err_cm:.1f} cm"
            )
            return (res.global_orient[-1], res.body_pose[-1], res.betas)
        except FitCancelled:
            return warm_start
        except Exception as exc:
            traceback.print_exc()
            self.state.status = (
                f"streaming... {n_frames_so_far} frames but SMPL-X fit failed for a chunk: "
                f"{type(exc).__name__}: {exc}"
            )
            return warm_start

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
            self.state.nodes.clear_body()
            self.state.nodes.clear_rest()
            if (
                self.state.stream_buffers.accum_joints is not None
                and self.state.nodes.motion is None
            ):
                self.state.nodes.motion = create_skeleton_sequence_node(
                    self.state.stream_buffers.accum_joints
                )
                self.state.nodes.motion.name = "Motion (streaming)"
                self.host.scene.add(self.state.nodes.motion)
            return
        if self.state.stream_buffers.accum_joints is None or self.state.smplx_fit.orient:
            self.schedule_remesh()
        else:
            self.__start_retrofit()

    def __start_retrofit(self) -> None:
        joints = self.state.stream_buffers.accum_joints
        if joints is None or self.state.generating:
            return
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
            target=self.__fit_existing_motion, args=(joints.copy(),), daemon=True
        ).start()

    def __fit_existing_motion(self, joints: np.ndarray) -> None:
        warm_start = None
        window = self.schedule.retrofit_window
        n_frames = 0
        try:
            for start in range(0, len(joints), window):
                if self.state.stream_buffers.cancel:
                    break
                piece = joints[start : start + window]
                n_frames += len(piece)
                warm_start = self.fit_single_chunk(piece, warm_start, n_frames)
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
        try:
            while self.state.smplx_fit.done_version != self.state.smplx_fit.version:
                version = self.state.smplx_fit.version
                gender = self.state.gender
                user = self.state.user_betas.copy()
                with self.state.stream_buffers.lock:
                    orient = (
                        np.concatenate(self.state.smplx_fit.orient, 0)
                        if self.state.smplx_fit.orient
                        else None
                    )
                    pose = (
                        np.concatenate(self.state.smplx_fit.pose, 0)
                        if self.state.smplx_fit.pose
                        else None
                    )
                    transl = (
                        np.concatenate(self.state.smplx_fit.transl, 0)
                        if self.state.smplx_fit.transl
                        else None
                    )
                    fit_betas = (
                        None
                        if self.state.smplx_fit.betas is None
                        else self.state.smplx_fit.betas.copy()
                    )
                cfg = replace(self.state.config.fit, model_dir=self.state.model_dir, gender=gender)
                try:
                    self.state.smplx_fit.note = f"updating avatar ({gender})..."
                    if orient is None:
                        zero = np.zeros((1, 3), dtype=np.float32)
                        verts, faces = smplx_mesh_from_fit_parameters(
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
                        if pose is None or transl is None:
                            raise ValueError("smplx fit lacks pose or translation")
                        betas = user if fit_betas is None else fit_betas + user
                        verts, faces = smplx_mesh_from_fit_parameters(
                            cfg,
                            orient,
                            pose,
                            transl,
                            betas.astype(np.float32),
                            gender,
                            self.state.device,
                        )
                        is_rest = False
                    with self.state.stream_buffers.lock:
                        self.state.stream_buffers.pending_remesh = (verts, faces, is_rest)
                    self.state.smplx_fit.note = ""
                except Exception as exc:
                    traceback.print_exc()
                    self.state.smplx_fit.note = f"avatar update failed: {type(exc).__name__}: {exc}"
                self.state.smplx_fit.done_version = version
        finally:
            self.state.smplx_fit.running = False
        if self.state.smplx_fit.done_version != self.state.smplx_fit.version:
            self.schedule_remesh()
