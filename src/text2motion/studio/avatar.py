from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import roma
import smplx
import torch

from text2motion.motion.model import SMPLX_MODEL_TYPE
from text2motion.motion.representation import JOINTS, kinematic_bones
from text2motion.studio.config import SmplxFitConfig
from text2motion.studio.contracts import ViewerHost

logger = logging.getLogger(__name__)


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


def upsample_pose_parameters(
    anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor == 1:
        return global_orient, body_pose, transl
    frames = global_orient.shape[0]
    rotations = np.concatenate([global_orient[:, None], body_pose.reshape(frames, -1, 3)], axis=1)
    start_orient, start_pose, start_transl = (
        (global_orient[0], body_pose[0], transl[0]) if anchor is None else anchor
    )
    start_rotations = np.concatenate([start_orient[None], start_pose.reshape(-1, 3)])[None]
    rotation_keys = torch.as_tensor(np.concatenate([start_rotations, rotations]))
    steps = torch.arange(1, factor + 1, dtype=torch.float32) / factor
    blended = roma.rotvec_slerp(rotation_keys[:-1], rotation_keys[1:], steps)
    blended = blended.transpose(0, 1).reshape(frames * factor, -1, 3).numpy()
    transl_keys = np.concatenate([start_transl[None], transl])
    weights = steps.numpy()[None, :, None]
    moved = transl_keys[:-1, None] + (transl_keys[1:, None] - transl_keys[:-1, None]) * weights
    return (
        blended[:, 0].astype(np.float32),
        blended[:, 1:].reshape(frames * factor, -1).astype(np.float32),
        moved.reshape(frames * factor, 3).astype(np.float32),
    )


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


def __start_parameters(
    model,
    target: torch.Tensor,
    cfg: SmplxFitConfig,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = target.shape[0]
    if warm_start is None:
        root_position = target[:, 0].detach()
        yaw = __yaw_init(model, target, root_position, device, cfg.yaw_candidates_deg)
        body_pose = torch.zeros(frames, 63, device=device, requires_grad=True)
        betas = torch.zeros(1, cfg.num_betas, device=device, requires_grad=True)
        return yaw.requires_grad_(True), body_pose, betas
    last_orient, last_pose, last_betas = warm_start
    orient = torch.tensor(last_orient, dtype=torch.float32, device=device).expand(frames, -1)
    pose = torch.tensor(last_pose, dtype=torch.float32, device=device).expand(frames, -1)
    betas = torch.tensor(last_betas, dtype=torch.float32, device=device).unsqueeze(0)
    return orient.clone().requires_grad_(True), pose.clone().requires_grad_(True), betas


def __is_report_step(step: int, steps: int, every: int) -> bool:
    return step % every == 0 or step == steps - 1


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
    global_orient, body_pose, betas = __start_parameters(model, target, cfg, warm_start, dev)

    def forward():
        return model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas.expand(t, -1),
        ).joints[:, :JOINTS]

    stage1 = torch.optim.Adam([global_orient, transl], lr=cfg.lr, weight_decay=0.0)
    for it in range(cfg.stage1_iters):
        stage1.zero_grad()
        ((forward() - target) ** 2).sum(-1).mean().backward()
        stage1.step()
        if on_progress is not None and __is_report_step(it, cfg.stage1_iters, cfg.progress_every):
            on_progress(it, total_iters)

    stage2_params = [global_orient, body_pose, transl] + ([] if warm_start is not None else [betas])
    stage2 = torch.optim.Adam(stage2_params, lr=cfg.lr, weight_decay=0.0)
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
        if on_progress is not None and __is_report_step(it, cfg.stage2_iters, cfg.progress_every):
            on_progress(cfg.stage1_iters + it, total_iters)
        if verbose and __is_report_step(it, cfg.stage2_iters, cfg.verbose_every):
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


SHORT_BONE_PARENT_PULL = {12: 1.0}


def bone_directions(
    rest: torch.Tensor, target: torch.Tensor, joint: int, children: dict[int, list[int]]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    frames = target.shape[0]
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
                torch.full((frames,), 1.0 if depth == 0 else 0.5, device=target.device)
            )
    return rest_pairs, target_pairs, weight_pairs


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
            rest_pairs, target_pairs, weight_pairs = bone_directions(rest, target, joint, children)
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
            joint_rotation = roma.rigid_vectors_registration(
                torch.stack(rest_pairs, 1),
                torch.stack(target_pairs, 1),
                torch.stack(weight_pairs, 1),
            )
            if not isinstance(joint_rotation, torch.Tensor):
                raise TypeError("the vector registration must return one rotation tensor")
            global_rotation.append(joint_rotation)
        local_rotation = [global_rotation[0]] + [
            global_rotation[parents[joint]].transpose(-1, -2) @ global_rotation[joint]
            for joint in range(1, JOINTS)
        ]
        axis_angle = roma.rotmat_to_rotvec(torch.stack(local_rotation, 1))
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
        factor: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        try:
            fitted = self.__fitted_chunk(joints, n_frames_so_far, factor)
        except Exception as exc:
            traceback.print_exc()
            self.state.status = (
                f"streaming... {n_frames_so_far} frames but SMPL-X fit failed for a chunk: "
                f"{type(exc).__name__}: {exc}"
            )
            return warm_start
        return warm_start if fitted is None else fitted

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

    def __fitted_chunk(
        self, joints: np.ndarray, n_frames_so_far: int, factor: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        gender = self.state.gender
        frames = joints.shape[0]
        if self.state.device == "cuda":
            torch.cuda.empty_cache()
        model = self.__body_model(frames, gender)
        if self.state.stream_buffers.fit_is_stale():
            return None
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
            return None
        duration_so_far = n_frames_so_far / self.host.playback_fps
        self.state.status = (
            f"{'streaming' if self.state.generating else 'fitting body'}... "
            f"{n_frames_so_far} frames (~{duration_so_far:.1f}s), "
            f"body fit {res.joint_err_cm:.1f} cm"
        )
        return (res.global_orient[-1], res.body_pose[-1], res.betas)

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
                warm_start = self.fit_single_chunk(piece, warm_start, n_frames, 1)
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
