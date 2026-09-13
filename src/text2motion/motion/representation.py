import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from text2motion.motion.kinematics import (
    Skeleton,
    qbetween_np,
    qinv,
    qinv_np,
    qmul_np,
    qrot,
    qrot_np,
    quaternion_to_cont6d,
    quaternion_to_cont6d_np,
)

DIM = 263
JOINTS = 22
FPS = 20

SLICES = {
    "root": slice(0, 4),
    "ric": slice(4, 67),
    "rot6d": slice(67, 193),
    "vel": slice(193, 259),
    "foot": slice(259, 263),
}
GEO_TERMS = tuple(SLICES)
FK_TERMS = ("fk_self", "fk_gt")


class RepresentationTrack(StrEnum):
    HML3D_263 = "hml3d263"
    SMPLX_168 = "smplx168"


class MotionSource(StrEnum):
    HUMANML3D = "humanml3d"
    AMASS = "amass"
    INTER_X = "inter-x"


RAW_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ]
)

KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

_LEG_JOINTS = (5, 8)
_RIGHT_FEET = [8, 11]
_LEFT_FEET = [7, 10]
_FACE_JOINTS = [2, 1, 17, 16]
_FOOT_VELOCITY_THRESHOLD = 0.002


@dataclass(frozen=True)
class FeatureParams:
    raw_offsets: np.ndarray
    chains: list[list[int]]
    target_offsets: torch.Tensor
    face_joints: list[int]
    left_feet: list[int]
    right_feet: list[int]
    leg_joints: tuple[int, int]
    joint_count: int
    foot_velocity_threshold: float


def target_offsets(reference_joints: np.ndarray) -> torch.Tensor:
    raw_offsets = torch.from_numpy(RAW_OFFSETS)
    example = torch.from_numpy(reference_joints).float()
    skeleton = Skeleton(raw_offsets, KINEMATIC_CHAINS, "cpu")
    return skeleton.get_offsets_joints(example[0])


def feature_params(offsets: torch.Tensor) -> FeatureParams:
    return FeatureParams(
        raw_offsets=RAW_OFFSETS,
        chains=KINEMATIC_CHAINS,
        target_offsets=offsets,
        face_joints=_FACE_JOINTS,
        left_feet=_LEFT_FEET,
        right_feet=_RIGHT_FEET,
        leg_joints=_LEG_JOINTS,
        joint_count=JOINTS,
        foot_velocity_threshold=_FOOT_VELOCITY_THRESHOLD,
    )


def _uniform_skeleton(positions: np.ndarray, params: FeatureParams) -> np.ndarray:
    raw_offsets = torch.from_numpy(params.raw_offsets)
    target_offset = params.target_offsets

    src_skel = Skeleton(raw_offsets, params.chains, "cpu")
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    first_leg_joint, second_leg_joint = params.leg_joints
    src_leg_len = np.abs(src_offset[first_leg_joint]).max() + np.abs(
        src_offset[second_leg_joint]
    ).max()
    tgt_leg_len = np.abs(tgt_offset[first_leg_joint]).max() + np.abs(
        tgt_offset[second_leg_joint]
    ).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    quat_params = src_skel.inverse_kinematics_np(positions, params.face_joints)

    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def encode_joints(positions: np.ndarray, params: FeatureParams) -> tuple[np.ndarray, ...]:
    foot_threshold = params.foot_velocity_threshold
    face_joints = params.face_joints
    left_feet, right_feet = params.left_feet, params.right_feet
    raw_offsets = torch.from_numpy(params.raw_offsets)

    positions = _uniform_skeleton(positions, params)

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = face_joints
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

    def detect_feet(positions: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        velocity_limit = np.array([threshold, threshold])

        left_x = (positions[1:, left_feet, 0] - positions[:-1, left_feet, 0]) ** 2
        left_y = (positions[1:, left_feet, 1] - positions[:-1, left_feet, 1]) ** 2
        left_z = (positions[1:, left_feet, 2] - positions[:-1, left_feet, 2]) ** 2
        left_contact = ((left_x + left_y + left_z) < velocity_limit).astype(np.float32)

        right_x = (positions[1:, right_feet, 0] - positions[:-1, right_feet, 0]) ** 2
        right_y = (positions[1:, right_feet, 1] - positions[:-1, right_feet, 1]) ** 2
        right_z = (positions[1:, right_feet, 2] - positions[:-1, right_feet, 2]) ** 2
        right_contact = ((right_x + right_y + right_z) < velocity_limit).astype(np.float32)
        return left_contact, right_contact

    feet_l, feet_r = detect_feet(positions, foot_threshold)

    def rotation_features(positions: np.ndarray) -> tuple[np.ndarray, ...]:
        skeleton = Skeleton(raw_offsets, params.chains, "cpu")
        quat_params = skeleton.inverse_kinematics_np(positions, face_joints, smooth_forward=True)

        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = rotation_features(positions)

    def root_relative_positions(positions: np.ndarray) -> np.ndarray:
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    positions = root_relative_positions(positions)

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


def _recover_root(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def recover_from_ric(data: torch.Tensor, joint_count: int) -> torch.Tensor:
    r_rot_quat, r_pos = _recover_root(data)
    positions = data[..., 4 : (joint_count - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def recover_from_rot(data: torch.Tensor, joint_count: int, skeleton: Skeleton) -> torch.Tensor:
    r_rot_quat, r_pos = _recover_root(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joint_count - 1) * 3
    end_indx = start_indx + (joint_count - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joint_count, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def normalization_stats(clips: list[np.ndarray], joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not clips:
        raise ValueError("normalization_stats received no clips")

    data = np.concatenate(clips, axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    expected_dim = 8 + (joint_count - 1) * 9 + joint_count * 3
    if std.shape[-1] != expected_dim:
        raise ValueError(
            f"feature dim {std.shape[-1]} != expected {expected_dim} for {joint_count=}"
        )

    ric_end = 4 + (joint_count - 1) * 3
    rot_end = 4 + (joint_count - 1) * 9
    vel_end = rot_end + joint_count * 3
    groups = (
        (0, 1),
        (1, 3),
        (3, 4),
        (4, ric_end),
        (ric_end, rot_end),
        (rot_end, vel_end),
        (vel_end, expected_dim),
    )
    for start, stop in groups:
        std[start:stop] = std[start:stop].mean()

    return mean, std


def recover_skeleton(feat263: np.ndarray | torch.Tensor) -> np.ndarray:
    data = (
        feat263
        if isinstance(feat263, torch.Tensor)
        else torch.as_tensor(feat263, dtype=torch.float32)
    )
    return recover_from_ric(data.float(), JOINTS).cpu().numpy()


class StreamingSkeletonRecovery:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.yaw = 0.0
        self.offset = np.zeros(2, dtype=np.float32)

    @staticmethod
    def _yaw_quat(yaw: float) -> torch.Tensor:
        quat = torch.zeros(4, dtype=torch.float32)
        quat[0] = math.cos(yaw)
        quat[2] = math.sin(yaw)
        return quat

    def __call__(self, feat263: np.ndarray | torch.Tensor) -> np.ndarray:
        data = (
            feat263.float()
            if isinstance(feat263, torch.Tensor)
            else torch.as_tensor(np.asarray(feat263), dtype=torch.float32)
        )
        local = recover_from_ric(data, JOINTS)

        carry = self._yaw_quat(self.yaw).expand(local.shape[:-1] + (4,))
        placed = qrot(qinv(carry), local)
        placed[..., 0] += float(self.offset[0])
        placed[..., 2] += float(self.offset[1])

        yaw_next = self.yaw + float(data[..., 0].sum())
        step = torch.zeros(3, dtype=torch.float32)
        step[0] = data[-1, 1]
        step[2] = data[-1, 2]
        step = qrot(qinv(self._yaw_quat(yaw_next)), step)

        root_last = placed[-1, 0]
        self.offset = np.array(
            [float(root_last[0] + step[0]), float(root_last[2] + step[2])], dtype=np.float32
        )
        self.yaw = yaw_next
        return placed.cpu().numpy()


def kinematic_bones() -> np.ndarray:
    bones = [
        [chain[i], chain[i + 1]] for chain in KINEMATIC_CHAINS for i in range(len(chain) - 1)
    ]
    return np.array(bones, dtype=np.int64)
