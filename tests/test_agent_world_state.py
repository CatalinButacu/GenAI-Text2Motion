"""WorldState behaviour tests: pose update, velocity finite-diff, action reset.

The world-state object is the only mutable surface between the runner and
the condition predicates. If pose extraction or per-action reset is wrong,
every distance/rotation/duration check downstream is wrong with it.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.modules.agent.world_state import WorldState
from src.shared.constants import SMPLX


def _make_frame(position: np.ndarray, yaw: float = 0.0) -> np.ndarray:
    """Build a fake 168-d SMPL-X frame with given root position + yaw proxy."""
    frame = np.zeros(SMPLX.pose_dim)
    frame[0:3] = [0.0, yaw, 0.0]  # axis-angle, yaw via y-axis (per our convention)
    frame[3:6] = position
    return frame


class TestWorldStateUpdate(unittest.TestCase):
    def test_update_pulls_position_and_yaw(self):
        w = WorldState()
        frame = _make_frame(np.array([1.0, 2.0, 3.0]), yaw=0.5)
        w.update_from_frame(frame)
        np.testing.assert_array_equal(w.position, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(w.heading_yaw, 0.5)
        self.assertEqual(w.frames_in_action, 1)

    def test_velocity_is_finite_difference(self):
        w = WorldState()
        f1 = _make_frame(np.array([0.0, 0.0, 0.0]))
        w.update_from_frame(f1, prev_position=None)
        # First-frame velocity should still be the zero default (no prev given)
        np.testing.assert_array_equal(w.velocity, [0.0, 0.0, 0.0])
        prev = w.position.copy()
        f2 = _make_frame(np.array([0.5, 0.0, 0.1]))
        w.update_from_frame(f2, prev_position=prev)
        np.testing.assert_allclose(w.velocity, [0.5, 0.0, 0.1])

    def test_reset_for_new_action_captures_yaw(self):
        w = WorldState()
        w.heading_yaw = 1.0
        w.frames_in_action = 25
        w.reset_for_new_action()
        self.assertEqual(w.yaw_at_action_start, 1.0)
        self.assertEqual(w.frames_in_action, 0)
        # Reset must NOT clear position -- carryover continuity demands it
        # stays consistent with the avatar's actual world location
        np.testing.assert_array_equal(w.position, [0.0, 0.0, 0.0])

    def test_distance_to_known_object(self):
        w = WorldState()
        w.add_scene_object("ball", np.array([3.0, 4.0, 0.0]))  # 5m away from origin
        self.assertAlmostEqual(w.distance_to("ball"), 5.0)
        w.position = np.array([3.0, 4.0, 0.0])
        self.assertAlmostEqual(w.distance_to("ball"), 0.0)

    def test_distance_to_unknown_raises(self):
        w = WorldState()
        with self.assertRaises(KeyError):
            w.distance_to("nonexistent_object")

    def test_rotated_since_start_is_absolute(self):
        w = WorldState()
        w.yaw_at_action_start = 0.0
        w.heading_yaw = np.radians(30.0)
        self.assertAlmostEqual(w.rotated_since_start_deg(), 30.0, places=4)
        # Sign-agnostic
        w.heading_yaw = -np.radians(30.0)
        self.assertAlmostEqual(w.rotated_since_start_deg(), 30.0, places=4)

    def test_bad_frame_shape_raises(self):
        w = WorldState()
        with self.assertRaises(ValueError):
            w.update_from_frame(np.zeros((3, 3)))  # not 1-D


if __name__ == "__main__":
    unittest.main()
