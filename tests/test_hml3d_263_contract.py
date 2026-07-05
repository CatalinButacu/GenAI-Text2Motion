import numpy as np
import torch

from text2motion.data.hml3d import param_util
from text2motion.data.hml3d.feature import (
    build_tgt_offsets,
    default_params,
    process_file,
    recover_from_ric,
)

JOINTS_NUM = 22
EXPECTED_DIM = 263


def _synthetic_motion(num_frames: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = param_util.t2m_raw_offsets.astype(np.float64) * 0.3
    joints = np.tile(base[None], (num_frames, 1, 1))
    for chain in param_util.t2m_kinematic_chain:
        for j in range(1, len(chain)):
            joints[:, chain[j]] += joints[:, chain[j - 1]]
    t = np.linspace(0, 1, num_frames)[:, None, None]
    joints = joints + 0.05 * np.sin(2 * np.pi * t) + 0.02 * rng.standard_normal(joints.shape)
    joints[:, 0, [0, 2]] += 0.1 * t[:, 0]  # root drifts in xz so velocities are nonzero
    return joints


def test_263_layout_and_recovery() -> None:
    motion = _synthetic_motion()
    tgt_offsets = build_tgt_offsets(motion)
    params = default_params(tgt_offsets)

    data, global_positions, local_positions, l_velocity = process_file(motion, params)

    assert data.shape[-1] == EXPECTED_DIM, data.shape
    assert data.shape[0] == motion.shape[0] - 1

    root = 1 + 2 + 1  # rot_vel + lin_vel_xz + root_y
    ric = (JOINTS_NUM - 1) * 3
    rot6d = (JOINTS_NUM - 1) * 6
    local_vel = JOINTS_NUM * 3
    foot = 4
    assert root + ric + rot6d + local_vel + foot == EXPECTED_DIM
    assert 8 + (JOINTS_NUM - 1) * 9 + JOINTS_NUM * 3 == EXPECTED_DIM

    foot_contacts = data[:, -foot:]
    assert set(np.unique(foot_contacts)).issubset({0.0, 1.0})

    rec = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), JOINTS_NUM)
    assert rec.shape == (1, data.shape[0], JOINTS_NUM, 3), rec.shape
    assert torch.isfinite(rec).all()


def test_param_constants_exact() -> None:
    assert param_util.face_joint_indx == [2, 1, 17, 16]
    assert param_util.fid_l == [7, 10]
    assert param_util.fid_r == [8, 11]
    assert (param_util.l_idx1, param_util.l_idx2) == (5, 8)
    assert param_util.joints_num == 22
    assert param_util.feet_threshold == 0.002
    assert param_util.t2m_raw_offsets.shape == (22, 3)
    assert param_util.t2m_kinematic_chain == [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]
