import numpy as np
import torch

from text2motion.motion import representation as param_util
from text2motion.motion.representation import (
    encode_joints,
    feature_params,
    recover_from_ric,
    target_offsets,
)

JOINTS_NUM = 22
EXPECTED_DIM = 263


def _synthetic_motion(num_frames: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = param_util.RAW_OFFSETS.astype(np.float64) * 0.3
    joints = np.tile(base[None], (num_frames, 1, 1))
    for chain in param_util.KINEMATIC_CHAINS:
        for j in range(1, len(chain)):
            joints[:, chain[j]] += joints[:, chain[j - 1]]
    t = np.linspace(0, 1, num_frames)[:, None, None]
    joints = joints + 0.05 * np.sin(2 * np.pi * t) + 0.02 * rng.standard_normal(joints.shape)
    joints[:, 0, [0, 2]] += 0.1 * t[:, 0]  # root drifts in xz so velocities are nonzero
    return joints


def test_263_layout_and_recovery() -> None:
    motion = _synthetic_motion()
    params = feature_params(target_offsets(motion))

    data, global_positions, local_positions, l_velocity = encode_joints(motion, params)

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
    params = feature_params(target_offsets(_synthetic_motion()))
    assert params.face_joints == [2, 1, 17, 16]
    assert params.left_feet == [7, 10]
    assert params.right_feet == [8, 11]
    assert params.leg_joints == (5, 8)
    assert param_util.JOINTS == 22
    assert params.foot_velocity_threshold == 0.002
    assert param_util.RAW_OFFSETS.shape == (22, 3)
    assert param_util.KINEMATIC_CHAINS == [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]
