import numpy as np
import pytest

from text2motion.evaluation.metrics import (
    diversity,
    frechet_gaussian_distance,
    mm_dist,
    motion_statistics,
    r_precision,
)
from text2motion.motion.representation import FPS

JOINTS = 22


def test_frechet_distance_of_shifted_unit_gaussians_is_the_squared_shift():
    identity = np.eye(2)
    distance = frechet_gaussian_distance(np.zeros(2), identity, np.array([3.0, 4.0]), identity)
    assert distance == pytest.approx(25.0)


def test_frechet_distance_of_scaled_gaussians_matches_the_closed_form():
    identity = np.eye(2)
    distance = frechet_gaussian_distance(np.zeros(2), 4.0 * identity, np.zeros(2), identity)
    assert distance == pytest.approx(2.0)


def test_mm_dist_is_the_mean_distance_of_matched_rows():
    text = np.zeros((2, 2))
    motion = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert mm_dist(text, motion) == pytest.approx(2.5)


def test_diversity_of_identical_features_is_zero():
    assert diversity(np.ones((10, 4)), num_pairs=50) == 0.0


def test_r_precision_counts_known_ranks_in_one_pool():
    text = np.array([[0.0], [10.0], [20.0], [30.0]])
    motion = np.array([[0.0], [10.0], [31.0], [19.0]])
    scores = r_precision(text, motion, pool_size=4, top_k=3)
    assert scores.tolist() == [0.5, 0.75, 1.0]


def test_constant_velocity_has_no_acceleration_and_a_known_foot_slide():
    step = 0.01
    joints = np.zeros((30, JOINTS, 3))
    joints[:, :, 0] = (np.arange(30) * step)[:, None]
    statistics = motion_statistics(joints)
    assert statistics["mean_acceleration_cm_per_frame2"] == pytest.approx(0.0, abs=1e-9)
    assert statistics["mean_jerk_cm_per_frame3"] == pytest.approx(0.0, abs=1e-9)
    assert statistics["root_teleports"] == 0.0
    assert statistics["foot_skate_m_per_s"] == pytest.approx(step * FPS)


def test_constant_acceleration_and_one_root_jump_are_measured():
    acceleration = 0.02
    joints = np.zeros((30, JOINTS, 3))
    joints[:, :, 0] = (0.5 * acceleration * np.arange(30) ** 2)[:, None]
    assert motion_statistics(joints)["mean_acceleration_cm_per_frame2"] == pytest.approx(2.0)
    jump = np.zeros((30, JOINTS, 3))
    jump[15:, 0, 0] = 1.0
    assert motion_statistics(jump)["root_teleports"] == 1.0
