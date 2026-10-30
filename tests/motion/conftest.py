import numpy as np
import pytest

from text2motion.motion import representation


@pytest.fixture
def synthetic_joints() -> np.ndarray:
    frames = 30
    generator = np.random.default_rng(0)
    joints = np.tile(representation.RAW_OFFSETS.astype(np.float64)[None] * 0.3, (frames, 1, 1))
    for chain in representation.KINEMATIC_CHAINS:
        for index in range(1, len(chain)):
            joints[:, chain[index]] += joints[:, chain[index - 1]]
    time = np.linspace(0, 1, frames)[:, None, None]
    joints = (
        joints + 0.05 * np.sin(2 * np.pi * time) + 0.02 * generator.standard_normal(joints.shape)
    )
    joints[:, 0, [0, 2]] += 0.1 * time[:, 0]
    return joints
