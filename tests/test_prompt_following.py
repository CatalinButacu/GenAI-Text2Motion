"""Tests for scripts/evaluation/eval_prompt_following.py"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.evaluation.eval_prompt_following import (
    CATEGORIES,
    aggregate_suite_results,
    evaluate_gate,
    load_suite,
    score_prompt_following,
)

SUITE_FILE = "data/eval/prompt_following_suite.json"


class FakeParsedScene:
    def __init__(self, actions=None, entities=None):
        self.actions = actions or []
        self.entities = entities or []


class FakeAction:
    def __init__(self, verb):
        self.verb = verb


class FakeEntity:
    def __init__(self, label):
        self.label = label


class TestScorePromptFollowing:
    def test_perfect_single_action(self):
        scene = FakeParsedScene(
            actions=[FakeAction("walk")],
            entities=[FakeEntity("person")],
        )
        entry = {
            "required_actions": ["walk"],
            "required_entities": ["person"],
            "forbidden_actions": [],
            "ordered": False,
            "category": "single_action",
        }
        result = score_prompt_following("a person walks forward", scene, entry)
        assert result["action_hit_rate"] == pytest.approx(1.0)
        assert result["entity_hit_rate"] == pytest.approx(1.0)
        assert result["forbidden_violation_rate"] == pytest.approx(0.0)
        assert result["combined_score"] == pytest.approx(1.0)

    def test_zero_hit_rate(self):
        scene = FakeParsedScene(
            actions=[FakeAction("run")],
            entities=[FakeEntity("ball")],
        )
        entry = {
            "required_actions": ["dance"],
            "required_entities": ["chair"],
            "forbidden_actions": [],
            "ordered": False,
            "category": "single_action",
        }
        result = score_prompt_following("a person dances", scene, entry)
        assert result["action_hit_rate"] == pytest.approx(0.0)
        assert result["entity_hit_rate"] == pytest.approx(0.0)

    def test_partial_hit_multi_step(self):
        scene = FakeParsedScene(
            actions=[FakeAction("walk")],
            entities=[FakeEntity("person")],
        )
        entry = {
            "required_actions": ["walk", "kick"],
            "required_entities": ["person", "ball"],
            "forbidden_actions": [],
            "ordered": True,
            "category": "multi_step",
        }
        result = score_prompt_following("a person walks and kicks", scene, entry)
        assert result["action_hit_rate"] == pytest.approx(0.5)
        assert result["entity_hit_rate"] == pytest.approx(0.5)
        assert result["order_accuracy"] == pytest.approx(0.0)

    def test_none_parsed_scene(self):
        entry = {
            "required_actions": ["walk"],
            "required_entities": ["person"],
            "forbidden_actions": [],
            "ordered": False,
            "category": "single_action",
        }
        result = score_prompt_following("a person walks", None, entry)
        assert result["action_hit_rate"] == pytest.approx(0.0)
        assert result["entity_hit_rate"] == pytest.approx(0.0)

    def test_empty_expected_returns_full_score(self):
        scene = FakeParsedScene()
        entry = {
            "required_actions": [],
            "required_entities": [],
            "forbidden_actions": [],
            "ordered": False,
            "category": "single_action",
        }
        result = score_prompt_following("anything", scene, entry)
        assert result["action_hit_rate"] == pytest.approx(1.0)
        assert result["entity_hit_rate"] == pytest.approx(1.0)

    def test_required_keys_present(self):
        scene = FakeParsedScene()
        entry = {
            "required_actions": ["walk"],
            "required_entities": [],
            "forbidden_actions": [],
            "ordered": False,
            "category": "single_action",
        }
        result = score_prompt_following("walk", scene, entry)
        for key in (
            "action_hit_rate",
            "entity_hit_rate",
            "action_precision",
            "entity_precision",
            "forbidden_violation_rate",
            "order_accuracy",
            "combined_score",
            "detected_actions",
            "detected_entities",
            "category",
        ):
            assert key in result

    def test_forbidden_violation_detected(self):
        scene = FakeParsedScene(actions=[FakeAction("run")], entities=[FakeEntity("person")])
        entry = {
            "required_actions": ["walk"],
            "required_entities": ["person"],
            "forbidden_actions": ["run"],
            "ordered": False,
            "category": "negation",
        }
        result = score_prompt_following("a person does not run, but walks instead", scene, entry)
        assert result["forbidden_violation_rate"] == pytest.approx(1.0)

    def test_order_accuracy_detected(self):
        scene = FakeParsedScene(actions=[FakeAction("walk"), FakeAction("kick")])
        entry = {
            "required_actions": ["walk", "kick"],
            "required_entities": ["person"],
            "forbidden_actions": [],
            "ordered": True,
            "category": "multi_step",
        }
        result = score_prompt_following("walk then kick", scene, entry)
        assert result["order_accuracy"] == pytest.approx(1.0)


class TestAggregateSuiteResults:
    def test_global_score_correct(self):
        results = [
            {
                "action_hit_rate": 1.0,
                "entity_hit_rate": 1.0,
                "combined_score": 1.0,
                "category": "single_action",
                "difficulty": "easy",
            },
            {
                "action_hit_rate": 0.0,
                "entity_hit_rate": 0.0,
                "combined_score": 0.0,
                "category": "single_action",
                "difficulty": "hard",
            },
        ]
        summary = aggregate_suite_results(results)
        assert summary["global_action_hit_rate"] == pytest.approx(0.5)
        assert summary["global_combined_score"] == pytest.approx(0.5)
        assert "by_difficulty" in summary
        assert summary["by_difficulty"]["easy"] == pytest.approx(1.0)
        assert summary["by_difficulty"]["hard"] == pytest.approx(0.0)

    def test_by_category_keys_present(self):
        results = []
        summary = aggregate_suite_results(results)
        for cat in CATEGORIES:
            assert cat in summary["by_category"]
            assert "combined_score" in summary["by_category"][cat]
            assert "forbidden_violation_rate" in summary["by_category"][cat]
            assert "order_accuracy" in summary["by_category"][cat]

    def test_empty_results(self):
        summary = aggregate_suite_results([])
        assert summary["n_prompts"] == 0
        assert summary["global_combined_score"] == pytest.approx(0.0)
        assert summary["global_forbidden_violation_rate"] == pytest.approx(0.0)
        assert summary["global_order_accuracy"] is None


class TestLoadSuite:
    def test_suite_file_exists_and_valid(self):
        if not Path(SUITE_FILE).exists():
            pytest.skip("Suite file not present")
        suite = load_suite(SUITE_FILE)
        assert len(suite) == 200
        for entry in suite:
            assert "id" in entry
            assert "prompt" in entry
            assert "category" in entry
            assert entry["category"] in CATEGORIES
            assert "required_actions" in entry
            assert "required_entities" in entry
            assert "forbidden_actions" in entry
            assert "ordered" in entry


class TestEvaluateGate:
    def test_gate_passes(self):
        summary = {
            "global_combined_score": 0.9,
            "global_forbidden_violation_rate": 0.05,
            "global_order_accuracy": 0.8,
            "by_category": {
                "single_action": {"combined_score": 0.8},
                "multi_step": {"combined_score": 0.82},
                "style_modifier": {"combined_score": 0.86},
                "negation": {"combined_score": 0.79},
                "object_interaction": {"combined_score": 0.81},
            },
        }
        gate = evaluate_gate(summary)
        assert gate["passed"] is True
        assert gate["failures"] == []

    def test_gate_fails_on_thresholds(self):
        summary = {
            "global_combined_score": 0.6,
            "global_forbidden_violation_rate": 0.2,
            "global_order_accuracy": 0.5,
            "by_category": {
                "single_action": {"combined_score": 0.6},
                "multi_step": {"combined_score": 0.8},
                "style_modifier": {"combined_score": 0.9},
                "negation": {"combined_score": 0.7},
                "object_interaction": {"combined_score": 0.9},
            },
        }
        gate = evaluate_gate(summary)
        assert gate["passed"] is False
        assert len(gate["failures"]) >= 3
