import numpy as np
import scipy.ndimage as ndimage
import torch


def qinv(q: torch.Tensor) -> torch.Tensor:
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qinv_np(q: np.ndarray) -> np.ndarray:
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return qinv(torch.from_numpy(q).float()).numpy()


def qnormalize(q: torch.Tensor) -> torch.Tensor:
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return q / torch.norm(q, dim=-1, keepdim=True)


def qmul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qmul_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qrot_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_np(quaternions: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d_np(quaternions: np.ndarray) -> np.ndarray:
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def quaternion_to_cont6d(quaternions: torch.Tensor) -> torch.Tensor:
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d


def cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def cont6d_to_matrix_np(cont6d: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(cont6d).contiguous().float()
    return cont6d_to_matrix(q).numpy()


def qbetween(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v = torch.cross(v0, v1)
    w = torch.sqrt((v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)) + (
        v0 * v1
    ).sum(dim=-1, keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))


def qbetween_np(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


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

    def get_offsets_joints(self, joints: torch.Tensor) -> torch.Tensor:
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    def inverse_kinematics_np(
        self,
        joints: np.ndarray,
        face_joint_idx: list[int],
        smooth_forward: bool = False,
    ) -> np.ndarray:
        assert len(face_joint_idx) == 4
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]

        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = ndimage.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

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

    def forward_kinematics_np(
        self,
        quat_params: np.ndarray,
        root_pos: np.ndarray,
        skel_joints: np.ndarray | None = None,
        do_root_R: bool = True,
    ) -> np.ndarray:
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
