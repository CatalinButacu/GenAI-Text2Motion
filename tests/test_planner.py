"""M2 - Planner module tests.

Uses synthetic ParsedScene objects (no file I/O, no checkpoints).
"""

from __future__ import annotations

import unittest

from src.modules.planner import ScenePlanner
from src.modules.planner.config import PlannerConfig
from src.modules.planner.constraint_layout import buildConstraints, solveLayout
from src.modules.planner.models import PlannedScene, Position3D
from src.modules.understanding.models import (
    ParsedAction,
    ParsedEntity,
    ParsedScene,
    SpatialRelation,
)


def _entity(name: str, objType: str, isActor: bool = False) -> ParsedEntity:
    return ParsedEntity(name=name, objectType=objType, isActor=isActor)


def _action(actionType: str, actor: str, target: str = "") -> ParsedAction:
    return ParsedAction(actionType=actionType, actor=actor, target=target)


def _scene(*entities, actions=(), spatial=()):
    for e in entities:
        for a in actions:
            if a.actor == e.name:
                e.actions.append(a)
    return ParsedScene(entities=list(entities), spatialRelations=list(spatial))


class TestPosition3D(unittest.TestCase):
    def test_to_list(self):
        self.assertEqual(Position3D(1.0, 2.0, 3.0).toList(), [1.0, 2.0, 3.0])

    def test_add(self):
        r = Position3D(1.0, 0.0, 0.0) + Position3D(0.0, 2.0, 0.0)
        self.assertEqual(r, Position3D(1.0, 2.0, 0.0))


class TestScenePlannerRow(unittest.TestCase):
    def setUp(self):
        self.planner = ScenePlanner(PlannerConfig())

    def test_single_actor_positioned(self):
        actor = _entity("person", "humanoid", isActor=True)
        scene = _scene(actor)
        planned = self.planner.plan(scene)
        self.assertEqual(len(planned.entities), 1)
        self.assertEqual(planned.entities[0].name, "person")

    def test_actor_and_object(self):
        actor = _entity("person", "humanoid", isActor=True)
        ball = _entity("sphere", "sphere")
        scene = _scene(actor, ball)
        planned = self.planner.plan(scene)
        self.assertEqual(len(planned.entities), 2)

    def test_fall_actor_elevated(self):
        ball = _entity("sphere", "sphere")
        a = _action("fall", "sphere")
        ball.actions.append(a)
        scene = _scene(ball)
        planned = self.planner.plan(scene)
        ball_planned = next(e for e in planned.entities if e.name == "sphere")
        self.assertGreater(ball_planned.position.z, 0.3)

    def test_duration_passed_through(self):
        scene = ParsedScene(entities=[], duration=5.0, durationExplicit=True)
        planned = self.planner.plan(scene)
        self.assertAlmostEqual(planned.duration, 5.0)

    def test_returns_planned_scene(self):
        scene = _scene(_entity("sphere", "sphere"))
        result = self.planner.plan(scene)
        self.assertIsInstance(result, PlannedScene)


class TestRandomLayout(unittest.TestCase):
    def test_random_positions_differ_from_row(self):
        actor = _entity("person", "humanoid", isActor=True)
        row_planner = ScenePlanner(PlannerConfig(randomLayout=False))
        rand_planner = ScenePlanner(PlannerConfig(randomLayout=True))
        scene = _scene(actor)
        row_pos = row_planner.plan(scene).entities[0].position
        rand_pos = rand_planner.plan(scene).entities[0].position
        # At minimum, one coordinate differs from row placement
        self.assertFalse(row_pos == rand_pos)


class TestConstraintLayout(unittest.TestCase):
    def test_solve_with_on_relation(self):
        entities = ["ball", "cube"]
        rel = SpatialRelation(subject="ball", predicate="on top of", relation="ON", object="cube")
        result = solveLayout(entities, [rel])
        self.assertIn("ball", result)
        self.assertIn("cube", result)

    def test_no_relations_returns_empty(self):
        result = solveLayout(["a", "b"], [])
        self.assertEqual(result, {})

    def test_build_constraints_filters_unknown(self):
        rels = [
            SpatialRelation(subject="a", predicate="near", relation="NEAR", object="b"),
            SpatialRelation(subject="", predicate="on", relation="ON", object="b"),  # empty subj
        ]
        constraints = buildConstraints(rels)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].ctype, "NEAR")


if __name__ == "__main__":
    unittest.main()
