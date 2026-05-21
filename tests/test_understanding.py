"""Understanding module tests.

All examples are people-centric: a single person, person-person, or
person-object interactions. No external files or checkpoints required.
SpacyParser is constructed once per class to amortise model loading.

Runs both via pytest and directly:
    pytest tests/test_understanding.py
    python tests/test_understanding.py
"""

from __future__ import annotations

import unittest

from src.modules.understanding.models import (
    ParsedAction,
    ParsedEntity,
    ParsedScene,
    SpatialRelation,
)
from src.modules.understanding.parsing_utils import SPATIAL_RELATIONS, split_into_clauses
from src.modules.understanding.spacy import SpacyParser


class TestModels(unittest.TestCase):
    def test_parsed_scene_actions_flat(self):
        e = ParsedEntity(name="humanoid", object_type="humanoid", is_actor=True)
        a = ParsedAction(action_type="walk", actor="humanoid")
        e.actions.append(a)
        scene = ParsedScene(prompt="a person walks", entities=[e])
        self.assertEqual(scene.actions, [a])

    def test_spatial_relation_fields(self):
        r = SpatialRelation(
            subject="humanoid_1",
            predicate="behind",
            relation="BEHIND",
            object="humanoid_2",
        )
        self.assertEqual(r.relation, "BEHIND")

    def test_slots_reject_extra_attr(self):
        e = ParsedEntity(name="humanoid", object_type="humanoid", is_actor=True)
        with self.assertRaises(AttributeError):
            e.nonexistent = 1  # type: ignore[attr-defined]

class TestParsingUtils(unittest.TestCase):
    def test_split_single_clause(self):
        clauses = split_into_clauses("a person walks")
        self.assertEqual(len(clauses), 1)
        self.assertFalse(clauses[0][1])

    def test_split_sequential(self):
        clauses = split_into_clauses("a person walks then runs")
        self.assertEqual(len(clauses), 2)

    def test_split_concurrent(self):
        clauses = split_into_clauses("a person walks while another person runs")
        self.assertTrue(clauses[1][1])

    def test_spatial_relations_has_canonical_keys(self):
        self.assertIn("in front of", SPATIAL_RELATIONS)
        self.assertEqual(SPATIAL_RELATIONS["in front of"], "IN_FRONT_OF")
        self.assertEqual(SPATIAL_RELATIONS["behind"], "BEHIND")

class TestSpacyParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = SpacyParser()

    def parse(self, text: str) -> ParsedScene:
        return self.parser.parse(text)

    def test_empty_prompt_returns_empty_scene(self):
        scene = self.parse("")
        self.assertEqual(scene.entities, [])
        self.assertEqual(scene.actions, [])

    def test_single_person_walking(self):
        scene = self.parse("a person walks")
        actors = [e for e in scene.entities if e.is_actor]
        self.assertEqual(len(actors), 1)
        self.assertEqual(actors[0].object_type, "humanoid")

    def test_action_extracted(self):
        scene = self.parse("a person walks forward")
        action_types = [a.action_type for a in scene.actions]
        self.assertIn("walk", action_types)

    def test_sequential_actions_one_person(self):
        scene = self.parse("a person walks then runs")
        by_type = {a.action_type: a.order for a in scene.actions}
        self.assertIn("walk", by_type)
        self.assertIn("run", by_type)
        self.assertLess(by_type["walk"], by_type["run"])

    def test_person_kicks_red_ball(self):
        scene = self.parse("a person kicks a red ball")
        humans = [e for e in scene.entities if e.object_type == "humanoid"]
        balls = [e for e in scene.entities if e.object_type == "sphere"]
        self.assertEqual(len(humans), 1)
        self.assertEqual(len(balls), 1)
        self.assertEqual(balls[0].skin, "red")
        kicks = [a for a in scene.actions if a.action_type == "kick"]
        self.assertEqual(len(kicks), 1)
        self.assertEqual(kicks[0].actor, humans[0].name)
        self.assertEqual(kicks[0].target, balls[0].name)

    def test_person_carries_book(self):
        scene = self.parse("a person carries a book")
        carries = [a for a in scene.actions if a.action_type == "carry"]
        self.assertEqual(len(carries), 1)
        self.assertNotEqual(carries[0].target, "")

    def test_two_people_fight(self):
        scene = self.parse("two people fight")
        humans = [e for e in scene.entities if e.object_type == "humanoid"]
        self.assertEqual(len(humans), 2)
        fights = [a for a in scene.actions if a.action_type == "fight"]
        fighters = {a.actor for a in fights}
        self.assertEqual(fighters, {humans[0].name, humans[1].name})

    def test_concurrent_two_people(self):
        scene = self.parse("a person waves while another person walks")
        humans = [e for e in scene.entities if e.object_type == "humanoid"]
        self.assertEqual(len(humans), 2)

        for a in scene.actions:
            self.assertEqual(a.order, 0)

    def test_duration_explicit(self):
        scene = self.parse("a person runs for 3 seconds")
        self.assertTrue(scene.duration_explicit)
        self.assertAlmostEqual(scene.duration, 3.0)

    def test_duration_implicit(self):
        scene = self.parse("a person walks")
        self.assertFalse(scene.duration_explicit)
        self.assertAlmostEqual(scene.duration, 5.0)

    def test_spatial_person_in_front_of_chair(self):
        scene = self.parse("a person stands in front of a chair")
        canonical_values = set(SPATIAL_RELATIONS.values())
        self.assertGreater(len(scene.spatial_relations), 0)

        for sr in scene.spatial_relations:
            self.assertIn(sr.relation, canonical_values)
        relations = {r.relation for r in scene.spatial_relations}
        self.assertIn("IN_FRONT_OF", relations)

    def test_person_falls(self):
        scene = self.parse("a person falls")
        action_types = [a.action_type for a in scene.actions]
        self.assertIn("fall", action_types)

if __name__ == "__main__":
    unittest.main()
