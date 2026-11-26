from __future__ import annotations

from types import MappingProxyType

import numpy as np
import roma
import torch

from text2motion.motion.representation import JOINTS
from text2motion.studio.contracts import SmplxFitResult

SHORT_BONE_PARENT_PULL = MappingProxyType({12: 1.0})


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


def joint_children(parents: list[int]) -> dict[int, list[int]]:
    return {joint: [c for c in range(JOINTS) if parents[c] == joint] for joint in range(JOINTS)}


def rest_joints(model, frames: int, device: str, zeros_betas: torch.Tensor) -> torch.Tensor:
    return model(
        global_orient=torch.zeros(frames, 3, device=device),
        body_pose=torch.zeros(frames, 63, device=device),
        transl=torch.zeros(frames, 3, device=device),
        betas=zeros_betas,
    ).joints[0, :JOINTS]


def add_parent_axis_pairs(
    rest_pairs: list[torch.Tensor],
    target_pairs: list[torch.Tensor],
    weight_pairs: list[torch.Tensor],
    parent_rotation: torch.Tensor,
    joint: int,
    parent_pull: float,
    device: str,
) -> None:
    frames = parent_rotation.shape[0]
    basis = torch.eye(3, device=device)
    for axis in range(3):
        rest_pairs.append(basis[axis].expand(frames, 3))
        target_pairs.append(parent_rotation @ basis[axis])
        weight_pairs.append(
            torch.full((frames,), SHORT_BONE_PARENT_PULL.get(joint, parent_pull), device=device)
        )


def register_joint_rotation(
    rest_pairs: list[torch.Tensor],
    target_pairs: list[torch.Tensor],
    weight_pairs: list[torch.Tensor],
    parent_rotation: torch.Tensor,
    joint: int,
    parent_pull: float,
    device: str,
) -> torch.Tensor:
    add_parent_axis_pairs(
        rest_pairs, target_pairs, weight_pairs, parent_rotation, joint, parent_pull, device
    )
    joint_rotation = roma.rigid_vectors_registration(
        torch.stack(rest_pairs, 1),
        torch.stack(target_pairs, 1),
        torch.stack(weight_pairs, 1),
    )
    if not isinstance(joint_rotation, torch.Tensor):
        raise TypeError("the vector registration must return one rotation tensor")
    return joint_rotation


def global_rotations(
    rest: torch.Tensor,
    target: torch.Tensor,
    parents: list[int],
    parent_pull: float,
    device: str,
) -> list[torch.Tensor]:
    frames = target.shape[0]
    children = joint_children(parents)
    identity = torch.eye(3, device=device).expand(frames, 3, 3)
    global_rotation: list[torch.Tensor] = []
    for joint in range(JOINTS):
        parent_rotation = identity if joint == 0 else global_rotation[parents[joint]]
        rest_pairs, target_pairs, weight_pairs = bone_directions(rest, target, joint, children)
        if not rest_pairs:
            global_rotation.append(parent_rotation)
            continue
        global_rotation.append(
            register_joint_rotation(
                rest_pairs, target_pairs, weight_pairs, parent_rotation, joint, parent_pull, device
            )
        )
    return global_rotation


def local_rotations(global_rotation: list[torch.Tensor], parents: list[int]) -> list[torch.Tensor]:
    return [global_rotation[0]] + [
        global_rotation[parents[joint]].transpose(-1, -2) @ global_rotation[joint]
        for joint in range(1, JOINTS)
    ]


def solve_smplx_pose_from_joints(
    model, target_joints: np.ndarray, device: str, parent_pull: float = 0.05
) -> SmplxFitResult:
    frames = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)
    zeros_betas = torch.zeros(frames, model.num_betas, device=device)
    with torch.no_grad():
        rest = rest_joints(model, frames, device, zeros_betas)
        parents = model.parents[:JOINTS].tolist()
        global_rotation = global_rotations(rest, target, parents, parent_pull, device)
        axis_angle = roma.rotmat_to_rotvec(
            torch.stack(local_rotations(global_rotation, parents), 1)
        )
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
