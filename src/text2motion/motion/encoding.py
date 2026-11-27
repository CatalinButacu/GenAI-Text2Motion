import numpy as np
import torch

from text2motion.motion.geometry import (
    qbetween_np,
    qinv_np,
    qmul_np,
    qrot_np,
    quaternion_to_cont6d_np,
)
from text2motion.motion.kinematics import Skeleton


def check_joint_shape(positions: np.ndarray) -> None:
    if positions.ndim != 3 or positions.shape[0] < 2:
        raise ValueError(
            f"encode_joints needs joints of shape [frames >= 2, joints, 3], got {positions.shape}"
        )


def lift_to_floor(positions: np.ndarray) -> np.ndarray:
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    return positions


def center_on_first_root(positions: np.ndarray) -> np.ndarray:
    root_pose_init_xz = positions[0][0] * np.array([1, 0, 1])
    return positions - root_pose_init_xz


def initial_heading_quat(
    root_pos_init: np.ndarray, face_joints: tuple[int, ...], shape: tuple[int, ...]
) -> np.ndarray:
    r_hip, l_hip, sdr_r, sdr_l = face_joints
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    return np.ones(shape + (4,)) * root_quat_init


def detect_foot_contacts(
    positions: np.ndarray,
    left_feet: tuple[int, ...],
    right_feet: tuple[int, ...],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
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


def rotation_features(
    positions: np.ndarray,
    raw_offsets: torch.Tensor,
    chains: tuple[tuple[int, ...], ...],
    face_joints: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    skeleton = Skeleton(raw_offsets, chains, "cpu")
    quat_params = skeleton.inverse_kinematics_np(positions, face_joints, smooth_forward=True)

    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    return cont_6d_params, r_velocity, velocity, r_rot


def root_relative_positions(positions: np.ndarray, r_rot: np.ndarray) -> np.ndarray:
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    return qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)


def root_channels(
    positions: np.ndarray, r_velocity: np.ndarray, velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    root_y = positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
    return root_data, l_velocity


def local_velocity(global_positions: np.ndarray, r_rot: np.ndarray) -> np.ndarray:
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    return local_vel.reshape(len(local_vel), -1)


def pack_features(
    root_data: np.ndarray,
    ric_data: np.ndarray,
    rot_data: np.ndarray,
    local_vel: np.ndarray,
    feet_l: np.ndarray,
    feet_r: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [root_data, ric_data[:-1], rot_data[:-1], local_vel, feet_l, feet_r], axis=-1
    )
