"""Pipeline integration tests.

Mocks all four module invoke() functions so no checkpoints, GPU, or heavy deps
are needed. Verifies the stage execution order and data passing.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.modules.motion.models import MotionClip
from src.modules.planner.models import PlannedScene
from src.modules.understanding.models import ParsedScene
from src.pipeline import Pipeline
from src.shared.config import PipelineConfig


def _mock_parsed() -> ParsedScene:
    return ParsedScene(prompt="a person walks", duration=4.0)


def _mock_planned() -> PlannedScene:
    return PlannedScene(entities=[], duration=4.0)


def _mock_clips() -> dict[str, MotionClip]:
    return {"person": MotionClip(action="walk", smplxParams=np.zeros((120, 168)))}


class TestPipelineStageOrder(unittest.TestCase):
    """Verify stages run in order and each stage gets the previous stage's output."""

    @patch("src.pipeline.render")
    @patch("src.pipeline.motion")
    @patch("src.pipeline.planner")
    @patch("src.pipeline.understanding")
    def test_stage_order(self, mockM1, mockM2, mockM4, mockRender):
        parsed = _mock_parsed()
        planned = _mock_planned()
        clips = _mock_clips()

        mockM1.invoke.return_value = parsed
        mockM2.invoke.return_value = planned
        mockM4.invoke.return_value = clips
        mockRender.invoke.return_value = "outputs/videos/output.mp4"

        pipeline = Pipeline(PipelineConfig())
        result = pipeline.run("a person walks", outputName="output")

        # Each stage was called exactly once
        mockM1.invoke.assert_called_once()
        mockM2.invoke.assert_called_once()
        mockM4.invoke.assert_called_once()
        mockRender.invoke.assert_called_once()

        # M2 receives M1 output
        self.assertIs(mockM2.invoke.call_args[0][0], parsed)

        # M4 receives M2 planned scene (not raw ParsedScene)
        self.assertIs(mockM4.invoke.call_args[0][0], planned)

        # Result dict contains expected keys
        self.assertEqual(result["parsed_scene"], parsed)
        self.assertEqual(result["planned_scene"], planned)
        self.assertEqual(result["motion_clips"], clips)

    @patch("src.pipeline.render")
    @patch("src.pipeline.motion")
    @patch("src.pipeline.planner")
    @patch("src.pipeline.understanding")
    def test_empty_prompt_returns_error(self, mockM1, mockM2, mockM4, mockRender):
        pipeline = Pipeline(PipelineConfig())
        result = pipeline.run("")
        self.assertIn("error", result)
        mockM1.invoke.assert_not_called()

    @patch("src.pipeline.render")
    @patch("src.pipeline.motion")
    @patch("src.pipeline.planner")
    @patch("src.pipeline.understanding")
    def test_result_contains_elapsed(self, mockM1, mockM2, mockM4, mockRender):
        mockM1.invoke.return_value = _mock_parsed()
        mockM2.invoke.return_value = _mock_planned()
        mockM4.invoke.return_value = _mock_clips()
        mockRender.invoke.return_value = "out.mp4"

        result = Pipeline(PipelineConfig()).run("a person walks")
        self.assertIn("elapsed_seconds", result)
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)


class TestPipelineConfig(unittest.TestCase):
    def test_video_path_format(self):
        cfg = PipelineConfig(outputDir="outputs")
        path = cfg.videoPath("my_test")
        self.assertIn("my_test.mp4", path)

    def test_duration_propagated_to_planner(self):
        cfg = PipelineConfig(duration=8.0)
        self.assertAlmostEqual(cfg.planner.baseDuration, 8.0)


if __name__ == "__main__":
    unittest.main()
