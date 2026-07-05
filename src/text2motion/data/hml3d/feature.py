from dataclasses import dataclass

import numpy as np
import torch

from . import param_util
from .quaternion import (
    qbetween_np,
    qinv,
    qinv_np,
    qmul_np,
    qrot,
    qrot_np,
    quaternion_to_cont6d,
    quaternion_to_cont6d_np,
)
from .skeleton import Skeleton


@dataclass(frozen=True)
class FeatureParams:
    n_raw_offsets: np.ndarray
    kinematic_chain: list[list[int]]
    tgt_offsets: torch.Tensor  # target skeleton offsets (joints_num, 3)
    face_joint_indx: list[int]
    fid_l: list[int]
    fid_r: list[int]
    l_idx1: int
    l_idx2: int
    joints_num: int
    feet_threshold: float


def build_tgt_offsets(reference_joints: np.ndarray) -> torch.Tensor:
    n_raw_offsets = torch.from_numpy(param_util.t2m_raw_offsets)
    example = torch.from_numpy(reference_joints).float()
    tgt_skel = Skeleton(n_raw_offsets, param_util.t2m_kinematic_chain, "cpu")
    return tgt_skel.get_offsets_joints(example[0])


def default_params(tgt_offsets: torch.Tensor) -> FeatureParams:
    return FeatureParams(
        n_raw_offsets=param_util.t2m_raw_offsets,
        kinematic_chain=param_util.t2m_kinematic_chain,
        tgt_offsets=tgt_offsets,
        face_joint_indx=param_util.face_joint_indx,
        fid_l=param_util.fid_l,
        fid_r=param_util.fid_r,
        l_idx1=param_util.l_idx1,
        l_idx2=param_util.l_idx2,
        joints_num=param_util.joints_num,
        feet_threshold=param_util.feet_threshold,
    )


def uniform_skeleton(positions: np.ndarray, params: FeatureParams) -> np.ndarray:
    n_raw_offsets = torch.from_numpy(params.n_raw_offsets)
    target_offset = params.tgt_offsets

    src_skel = Skeleton(n_raw_offsets, params.kinematic_chain, "cpu")
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    src_leg_len = np.abs(src_offset[params.l_idx1]).max() + np.abs(src_offset[params.l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[params.l_idx1]).max() + np.abs(tgt_offset[params.l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    quat_params = src_skel.inverse_kinematics_np(positions, params.face_joint_indx)

    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions: np.ndarray, params: FeatureParams) -> tuple[np.ndarray, ...]:
    feet_thre = params.feet_threshold
    face_joint_indx = params.face_joint_indx
    fid_l, fid_r = params.fid_l, params.fid_r
    n_raw_offsets = torch.from_numpy(params.n_raw_offsets)
    kinematic_chain = params.kinematic_chain

    positions = uniform_skeleton(positions, params)

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    global_positions = positions.copy()

    def foot_detect(positions: np.ndarray, thres: float) -> tuple[np.ndarray, np.ndarray]:
        velfactor = np.array([thres, thres])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    def get_cont6d_params(positions: np.ndarray) -> tuple[np.ndarray, ...]:
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)

    def get_rifke(positions: np.ndarray) -> np.ndarray:
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    positions = get_rifke(positions)

    root_y = positions[:, 0, 1:2]

    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    ric_data = positions[:, 1:].reshape(len(positions), -1)

    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


def recover_root_rot_pos(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def recover_from_rot(data: torch.Tensor, joints_num: int, skeleton: Skeleton) -> torch.Tensor:
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions
