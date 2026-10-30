import numpy as np
import pytest
import torch

from text2motion.motion import representation
from text2motion.motion.kinematics import Skeleton
from text2motion.motion.preparation import swap_left_right


def __encoded(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params = representation.feature_params(representation.target_offsets(joints))
    features, global_positions, _, _ = representation.encode_joints(joints, params)
    return features, global_positions


def test_recovery_from_ric_reproduces_the_encoded_global_joints(synthetic_joints):
    features, global_positions = __encoded(synthetic_joints)

    recovered = representation.recover_from_ric(torch.from_numpy(features).float(), 22)

    np.testing.assert_allclose(recovered.numpy(), global_positions[:-1], atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 3, 7, 16])
def test_streamed_recovery_equals_whole_recovery(synthetic_joints, chunk):
    features, _ = __encoded(synthetic_joints)
    whole = representation.recover_from_ric(torch.from_numpy(features).float(), 22).numpy()
    recovery = representation.StreamingSkeletonRecovery()

    parts = [recovery(features[start : start + chunk]) for start in range(0, len(features), chunk)]

    np.testing.assert_allclose(np.concatenate(parts), whole, atol=1e-5)


def test_batched_rotation_recovery_equals_recovery_of_each_sequence(synthetic_joints):
    features, global_positions = __encoded(synthetic_joints)
    skeleton = Skeleton(
        torch.tensor(representation.RAW_OFFSETS).float(), representation.KINEMATIC_CHAINS, "cpu"
    )
    skeleton.get_offsets_joints(torch.from_numpy(global_positions[0]).float())
    single = torch.from_numpy(features).float()
    one = representation.recover_from_rot(single, 22, skeleton)

    batched = representation.recover_from_rot(torch.stack([single, single * 0.5]), 22, skeleton)

    torch.testing.assert_close(batched[0], one)


@pytest.mark.parametrize("joint_count", [22, 52])
def test_mirroring_twice_gives_back_the_input(joint_count):
    joints = np.random.default_rng(1).standard_normal((5, joint_count, 3))

    np.testing.assert_array_equal(swap_left_right(swap_left_right(joints)), joints)


def test_the_shared_offsets_are_read_only_and_a_skeleton_keeps_its_own_copy(synthetic_joints):
    assert not representation.RAW_OFFSETS.flags.writeable
    source = representation.RAW_OFFSETS.astype(np.float32)
    skeleton = Skeleton(torch.from_numpy(source), representation.KINEMATIC_CHAINS, "cpu")
    before = skeleton.inverse_kinematics_np(synthetic_joints, (2, 1, 17, 16)).copy()
    assert np.isfinite(before).all()

    source[:] = 0.0

    after = skeleton.inverse_kinematics_np(synthetic_joints, (2, 1, 17, 16))
    np.testing.assert_array_equal(after, before)
