from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import roma
import smplx
import torch

from text2motion.motion.contracts import SMPLX_MODEL_TYPE
from text2motion.motion.representation import JOINTS
from text2motion.studio.config import SmplxFitConfig
from text2motion.studio.contracts import SmplxFitResult

logger = logging.getLogger(__name__)


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


SHORT_BONE_PARENT_PULL = MappingProxyType({12: 1.0})


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
