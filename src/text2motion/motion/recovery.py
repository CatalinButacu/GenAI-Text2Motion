import math

import numpy as np
import torch

from text2motion.motion.geometry import qinv, qrot, quaternion_to_cont6d
from text2motion.motion.kinematics import Skeleton
from text2motion.motion.representation import JOINTS


def check_feature_width(data: torch.Tensor, joint_count: int) -> None:
    def feature_width(joint_count: int) -> int:
        return 8 + (joint_count - 1) * 9 + joint_count * 3

    expected = feature_width(joint_count)
    if data.dim() < 2 or data.shape[-1] != expected:
        raise ValueError(
            f"recovery needs features of shape [..., frames, {expected}] for {joint_count} "
            f"joints, got {tuple(data.shape)}"
        )


def check_finite_features(data: torch.Tensor) -> None:
    if not torch.isfinite(data).all():
        raise ValueError("the motion features hold NaN or Inf values")


def recover_root(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), dtype=data.dtype, device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), dtype=data.dtype, device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joint_count: int) -> torch.Tensor:
    check_feature_width(data, joint_count)
    r_rot_quat, r_pos = recover_root(data)
    positions = data[..., 4 : (joint_count - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    return torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)


def recover_from_rot(data: torch.Tensor, joint_count: int, skeleton: Skeleton) -> torch.Tensor:
    check_feature_width(data, joint_count)
    r_rot_quat, r_pos = recover_root(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joint_count - 1) * 3
    end_indx = start_indx + (joint_count - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joint_count, 6)

    joints = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos.reshape(-1, 3))
    return joints.reshape(data.shape[:-1] + (joint_count, 3))


def recover_skeleton(feat263: np.ndarray | torch.Tensor) -> np.ndarray:
    data = (
        feat263
        if isinstance(feat263, torch.Tensor)
        else torch.as_tensor(feat263, dtype=torch.float32)
    )
    check_finite_features(data)
    return recover_from_ric(data.float(), JOINTS).cpu().numpy()


############################################################
#       StreamingSkeletonRecovery
############################################################
class StreamingSkeletonRecovery:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.yaw = 0.0
        self.offset = np.zeros(2, dtype=np.float32)

    @staticmethod
    def yaw_quat(yaw: float, device: torch.device) -> torch.Tensor:
        quat = torch.zeros(4, dtype=torch.float32, device=device)
        quat[0] = math.cos(yaw)
        quat[2] = math.sin(yaw)
        return quat

    def __call__(self, feat263: np.ndarray | torch.Tensor) -> np.ndarray:
        data = (
            feat263.float()
            if isinstance(feat263, torch.Tensor)
            else torch.as_tensor(np.asarray(feat263), dtype=torch.float32)
        )
        check_finite_features(data)
        local = recover_from_ric(data, JOINTS)

        carry = self.yaw_quat(self.yaw, data.device).expand(local.shape[:-1] + (4,))
        placed = qrot(qinv(carry), local)
        placed[..., 0] += float(self.offset[0])
        placed[..., 2] += float(self.offset[1])

        yaw_next = self.yaw + float(data[..., 0].sum())
        step = torch.zeros(3, dtype=torch.float32, device=data.device)
        step[0] = data[-1, 1]
        step[2] = data[-1, 2]
        step = qrot(qinv(self.yaw_quat(yaw_next, data.device)), step)

        root_last = placed[-1, 0]
        self.offset = np.array(
            [float(root_last[0] + step[0]), float(root_last[2] + step[2])], dtype=np.float32
        )
        self.yaw = yaw_next
        return placed.cpu().numpy()
