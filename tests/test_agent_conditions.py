"""Predicate grammar test suite for src/modules/agent/conditions.py.

Tests every branch of :func:`parse_condition` plus runtime evaluation
against a :class:`WorldState`. The whole point of the agent layer is
that the planner LM emits termination predicates the runner can trust;
if parsing silently accepts garbage or returns wrong-typed callables,
the closed loop fails opaquely at inference time.

Run with the rest: ``python -m pytest tests/test_agent_conditions.py``
"""
from __future__ import annotations

import unittest

import numpy as np

from src.modules.agent.conditions import parse_condition
from src.modules.agent.world_state import WorldState


def _world_with_tree(distance: float = 5.0) -> WorldState:
    w = WorldState()
    w.add_scene_object("tree", np.array([distance, 0.0, 0.0]))

    return w


class TestParseCondition(unittest.TestCase):

    def test_completed(self):
        cond = parse_condition("completed")
        self.assertEqual(cond.source, "completed")
        # Always fires
        self.assertTrue(cond(_world_with_tree()))

    def test_duration_fires_only_after_threshold(self):
        cond = parse_condition("duration(10)")
        w = _world_with_tree()
        w.frames_in_action = 9
        self.assertFalse(cond(w))
        w.frames_in_action = 10
        self.assertTrue(cond(w))

    def test_distance_lt(self):
        cond = parse_condition("distance(tree) < 1.0")
        w = _world_with_tree(distance=5.0)
        self.assertFalse(cond(w))
        # Move avatar close
        w.position = np.array([4.5, 0.0, 0.0])
        self.assertTrue(cond(w))

    def test_distance_gt(self):
        cond = parse_condition("distance(tree) > 10.0")
        w = _world_with_tree(distance=5.0)
        self.assertFalse(cond(w))
        # Walk away
        w.position = np.array([-6.0, 0.0, 0.0])  # 11 m from tree
        self.assertTrue(cond(w))

    def test_rotated(self):
        cond = parse_condition("rotated(90)")
        w = _world_with_tree()
        w.yaw_at_action_start = 0.0
        w.heading_yaw = np.radians(45.0)
        self.assertFalse(cond(w))
        w.heading_yaw = np.radians(90.0)
        self.assertTrue(cond(w))
        # Negative rotations also count -- abs()
        w.heading_yaw = -np.radians(95.0)
        self.assertTrue(cond(w))

    def test_unknown_object_raises_at_evaluation(self):
        cond = parse_condition("distance(unicorn) < 1.0")
        w = _world_with_tree()

        with self.assertRaises(KeyError):
            cond(w)

    def test_invalid_grammar_rejected(self):
        bad_inputs = [
            "",
            "walk forward",          # action, not a condition
            "until reach tree",       # English, not the grammar
            "distance(tree)",         # missing operator/threshold
            "duration",               # missing arg
            "rotated()",              # empty arg
            "distance(tree) <= 1.0",  # <= not in grammar (intentional)
        ]

        for inp in bad_inputs:
            with self.assertRaises(ValueError, msg=f"unexpected accept of {inp!r}"):
                parse_condition(inp)

    def test_whitespace_tolerance(self):
        # The planner may emit slightly different spacing; the parser should
        # accept anything that's structurally correct.
        for variant in [
            "duration(10)",
            "duration( 10 )",
            "distance(tree)<1.0",
            "distance( tree ) < 1.0",
            "  completed  ",
        ]:
            try:
                parse_condition(variant)
            except ValueError as e:
                self.fail(f"whitespace variant {variant!r} rejected: {e}")


if __name__ == "__main__":
    unittest.main()
