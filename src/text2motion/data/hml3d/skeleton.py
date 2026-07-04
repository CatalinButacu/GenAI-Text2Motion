"""Skeleton forward/inverse kinematics -- faithful port of the official HumanML3D code.

Ported verbatim (math unchanged) from:
    https://github.com/EricGuo5513/HumanML3D/blob/main/common/skeleton.py

The ONLY changes versus the original are cosmetic: explicit imports instead of
``from common.quaternion import *``, and dropping the unused euler/quaternion-fitting
helpers the 263 pipeline never calls. The numerical behaviour of the kept methods
(offsets, IK, FK, cont6d FK) is identical -- these must match the Guo evaluator's data.
"""

import numpy as np
import scipy.ndimage as ndimage
import torch

from .quaternion import (
    cont6d_to_matrix,
    cont6d_to_matrix_np,
    qbetween_np,
    qinv_np,
    qmul_np,
    qrot_np,
)


class Skeleton:
    def __init__(self, offset: torch.Tensor, kinematic_tree: list[list[int]], device: str) -> None:
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def njoints(self) -> int:
        return len(self._raw_offset)

    def offset(self) -> torch.Tensor | None:
        return self._offset

    def set_offset(self, offsets: torch.Tensor) -> None:
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self) -> list[list[int]]:
        return self._kinematic_tree

    def parents(self) -> list[int]:
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints: torch.Tensor) -> torch.Tensor:
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = (
                torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None]
                * _offsets[:, i]
            )

        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints: torch.Tensor) -> torch.Tensor:
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[i] = (
                torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]
            )

        self._offset = _offsets.detach()
        return _offsets

    # face_joint_idx order: right hip, left hip, right shoulder, left shoulder
    # joints (batch_size, joints_num, 3)
    def inverse_kinematics_np(
        self,
        joints: np.ndarray,
        face_joint_idx: list[int],
        smooth_forward: bool = False,
    ) -> np.ndarray:
        assert len(face_joint_idx) == 4
        # Get Forward Direction
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = ndimage.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        # Get Root Rotation
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        # Inverse Kinematics -- quat_params (batch_size, joints_num, 4)
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                u = self._raw_offset_np[chain[j + 1]][np.newaxis, ...].repeat(len(joints), axis=0)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(
        self,
        quat_params: np.ndarray,
        root_pos: np.ndarray,
        skel_joints: np.ndarray | None = None,
        do_root_R: bool = True,
    ) -> np.ndarray:
        # quat_params (batch_size, joints_num, 4); root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d(
        self,
        cont6d_params: torch.Tensor,
        root_pos: torch.Tensor,
        skel_joints: torch.Tensor | None = None,
        do_root_R: bool = True,
    ) -> torch.Tensor:
        # cont6d_params (batch_size, joints_num, 6); root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = (
                    torch.eye(3)
                    .expand((len(cont6d_params), -1, -1))
                    .detach()
                    .to(cont6d_params.device)
                )
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                joints[:, chain[i]] = (
                    torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
                )
        return joints

    def forward_kinematics_cont6d_np(
        self,
        cont6d_params: np.ndarray,
        root_pos: np.ndarray,
        skel_joints: np.ndarray | None = None,
        do_root_R: bool = True,
    ) -> np.ndarray:
        # cont6d_params (batch_size, joints_num, 6); root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                joints[:, chain[i]] = (
                    np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
                )
        return joints
