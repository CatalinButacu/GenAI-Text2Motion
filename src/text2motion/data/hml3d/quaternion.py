"""Quaternion / continuous-6D rotation math -- faithful port of the official HumanML3D code.

Ported verbatim (math unchanged) from:
    https://github.com/EricGuo5513/HumanML3D/blob/main/common/quaternion.py
which itself derives from Facebook's QuaterNet
    (Copyright (c) 2018-present, Facebook, Inc.; licensed under the QuaterNet LICENSE).

Fidelity is mandatory: these are the exact formulas the Guo evaluator and T2M-GPT's VQ-VAE
expect. The ONLY changes versus the original are cosmetic (PEP8 names kept as-is for the
public API the pipeline calls, explicit imports instead of ``import *``, and replacing the
removed ``np.float`` alias with ``np.float32``/``float``). No numerical behaviour changed.
"""

import numpy as np
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
    """Multiply quaternion(s) q with quaternion(s) r.

    Expects two equally-sized tensors of shape (*, 4). Returns q*r of shape (*, 4).
    """
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
    """Rotate vector(s) v by quaternion(s) q. q is (*, 4), v is (*, 3); returns (*, 3)."""
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


def qfix(q: np.ndarray) -> np.ndarray:
    """Enforce quaternion continuity across time by flipping sign to maximise consecutive dot.

    Expects (L, J, 4); returns the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Quaternions (real part first), shape (..., 4) -> rotation matrices (..., 3, 3)."""
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
    """Find the quaternion that rotates v0 to v1. Both are (*, 3)."""
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v = torch.cross(v0, v1)
    w = torch.sqrt(
        (v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)
    ) + (v0 * v1).sum(dim=-1, keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))


def qbetween_np(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Find the quaternion that rotates v0 to v1. Both are (*, 3)."""
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()
