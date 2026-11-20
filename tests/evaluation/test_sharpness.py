import numpy as np
import pytest

from text2motion.evaluation.metrics import paired_mean_ratio, sharpness_statistics
from text2motion.motion.representation import FPS, JOINTS


def test_a_joint_at_constant_speed_has_that_peak_speed_and_no_acceleration():
    frames = 40
    joints = np.zeros((frames, JOINTS, 3))
    joints[:, :, 0] = np.arange(frames)[:, None] * 0.5
    joints[:, 15, 1] = np.arange(frames) * (1.2 / FPS)
    statistics = sharpness_statistics(joints)
    assert statistics["peak_speed_m_per_s"] == pytest.approx(1.2)
    assert statistics["peak_acceleration_m_per_s2"] == pytest.approx(0.0, abs=1e-9)
    assert statistics["mean_jerk_m_per_s3"] == pytest.approx(0.0, abs=1e-9)


def test_the_paired_mean_ratio_of_a_doubled_sample_is_two():
    values = np.random.default_rng(2026).uniform(1.0, 2.0, 64)
    ratio, low, high = paired_mean_ratio(2.0 * values, values, 200, 2026)
    assert ratio == pytest.approx(2.0)
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(2.0)
