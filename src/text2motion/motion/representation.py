from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np
import torch

from text2motion.motion.encoding import (
    center_on_first_root,
    check_joint_shape,
    detect_foot_contacts,
    initial_heading_quat,
    lift_to_floor,
    local_velocity,
    pack_features,
    root_channels,
    root_relative_positions,
    rotation_features,
)
from text2motion.motion.geometry import qrot_np
from text2motion.motion.kinematics import Skeleton

DIM = 263
JOINTS = 22
FPS = 20

SLICES = MappingProxyType(
    {
        "root": slice(0, 4),
        "ric": slice(4, 67),
        "rot6d": slice(67, 193),
        "vel": slice(193, 259),
        "foot": slice(259, 263),
    }
)
GEO_TERMS = tuple(SLICES)
FK_TERMS = ("fk_self", "fk_gt")
DYNAMICS_TERMS = ("foot_skate", "accel")
FOOT_CONTACT_JOINTS = (7, 10, 8, 11)


############################################################
#       RepresentationTrack
############################################################
class RepresentationTrack(StrEnum):
    HML3D_263 = "hml3d263"
    SMPLX_168 = "smplx168"


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
RAW_OFFSETS.setflags(write=False)

KINEMATIC_CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)

_LEG_JOINTS = (5, 8)
_RIGHT_FEET = (8, 11)
_LEFT_FEET = (7, 10)
_FACE_JOINTS = (2, 1, 17, 16)
_FOOT_VELOCITY_THRESHOLD = 0.002
_RIGHT_CHAIN = (2, 5, 8, 11, 14, 17, 19, 21)
_LEFT_CHAIN = (1, 4, 7, 10, 13, 16, 18, 20)
_LEFT_HAND_CHAIN = (22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30)
_RIGHT_HAND_CHAIN = (43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51)


############################################################
#       FeatureParams
############################################################
@dataclass(frozen=True)
class FeatureParams:
    raw_offsets: np.ndarray
    chains: tuple[tuple[int, ...], ...]
    target_offsets: torch.Tensor
    face_joints: tuple[int, ...]
    left_feet: tuple[int, ...]
    right_feet: tuple[int, ...]
    leg_joints: tuple[int, int]
    joint_count: int
    foot_velocity_threshold: float


def swap_left_right(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    tmp = data[:, _RIGHT_CHAIN]
    data[:, _RIGHT_CHAIN] = data[:, _LEFT_CHAIN]
    data[:, _LEFT_CHAIN] = tmp
    if data.shape[1] > 24:
        tmp = data[:, _RIGHT_HAND_CHAIN]
        data[:, _RIGHT_HAND_CHAIN] = data[:, _LEFT_HAND_CHAIN]
        data[:, _LEFT_HAND_CHAIN] = tmp
    return data


def target_offsets(reference_joints: np.ndarray) -> torch.Tensor:
    raw_offsets = torch.tensor(RAW_OFFSETS)
    example = torch.from_numpy(reference_joints).float()
    skeleton = Skeleton(raw_offsets, KINEMATIC_CHAINS, "cpu")
    return skeleton.offsets.get_offsets_joints(example[0])


def feature_params(offsets: torch.Tensor, fps: int = FPS) -> FeatureParams:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return FeatureParams(
        raw_offsets=RAW_OFFSETS,
        chains=KINEMATIC_CHAINS,
        target_offsets=offsets,
        face_joints=_FACE_JOINTS,
        left_feet=_LEFT_FEET,
        right_feet=_RIGHT_FEET,
        leg_joints=_LEG_JOINTS,
        joint_count=JOINTS,
        foot_velocity_threshold=_FOOT_VELOCITY_THRESHOLD * (FPS / fps) ** 2,
    )


def uniform_skeleton(positions: np.ndarray, params: FeatureParams) -> np.ndarray:
    raw_offsets = torch.tensor(params.raw_offsets)
    target_offset = params.target_offsets

    src_skel = Skeleton(raw_offsets, params.chains, "cpu")
    src_offset = src_skel.offsets.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    first_leg_joint, second_leg_joint = params.leg_joints
    src_leg_len = (
        np.abs(src_offset[first_leg_joint]).max() + np.abs(src_offset[second_leg_joint]).max()
    )
    tgt_leg_len = (
        np.abs(tgt_offset[first_leg_joint]).max() + np.abs(tgt_offset[second_leg_joint]).max()
    )

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    quat_params = src_skel.inverse_kinematics_np(positions, params.face_joints)

    src_skel.offsets.set_offset(target_offset)
    return src_skel.forward_kinematics_np(quat_params, tgt_root_pos)


def encode_joints(positions: np.ndarray, params: FeatureParams) -> tuple[np.ndarray, ...]:
    check_joint_shape(positions)
    positions = uniform_skeleton(positions, params)
    positions = lift_to_floor(positions)

    root_pos_init = positions[0]
    positions = center_on_first_root(positions)
    root_quat_init = initial_heading_quat(root_pos_init, params.face_joints, positions.shape[:-1])
    positions = qrot_np(root_quat_init, positions)

    global_positions = positions.copy()
    feet_l, feet_r = detect_foot_contacts(
        positions, params.left_feet, params.right_feet, params.foot_velocity_threshold
    )
    cont_6d_params, r_velocity, velocity, r_rot = rotation_features(
        positions, torch.tensor(params.raw_offsets), params.chains, params.face_joints
    )
    positions = root_relative_positions(positions, r_rot)

    root_data, l_velocity = root_channels(positions, r_velocity, velocity)
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = positions[:, 1:].reshape(len(positions), -1)
    local_vel = local_velocity(global_positions, r_rot)
    data = pack_features(root_data, ric_data, rot_data, local_vel, feet_l, feet_r)
    return data, global_positions, positions, l_velocity


def kinematic_bones() -> np.ndarray:
    bones = [[chain[i], chain[i + 1]] for chain in KINEMATIC_CHAINS for i in range(len(chain) - 1)]
    return np.array(bones, dtype=np.int64)
