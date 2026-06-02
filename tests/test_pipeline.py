from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.modules.motion.models import MotionClip
from src.modules.planner.models import PlannedScene
from src.modules.understanding.models import ParsedScene
from src.pipeline import Pipeline
from src.shared.config import PipelineConfig


def mock_parsed() -> ParsedScene:
    return ParsedScene(prompt="a person walks", duration=4.0)


def mock_planned() -> PlannedScene:
    return PlannedScene(entities=[], duration=4.0)


def mock_clips() -> dict[str, MotionClip]:
    return {"person": MotionClip(action="walk", smplx_params=np.zeros((120, 168)))}


def pipeline_with_injected_stages(parsed, planned, clips) -> Pipeline:
    pipeline = Pipeline(PipelineConfig())
    pipeline.parser = MagicMock()
    pipeline.parser.parse.return_value = parsed
    pipeline.layout = MagicMock()
    pipeline.layout.plan.return_value = planned
    pipeline.motion = MagicMock()
    pipeline.motion.generate_for_scene.return_value = clips
    return pipeline


class TestPipelineStageOrder(unittest.TestCase):
    @patch("src.pipeline.render_clip_to_file")
    def test_stage_order(self, mock_render):
        parsed, planned, clips = mock_parsed(), mock_planned(), mock_clips()
        pipeline = pipeline_with_injected_stages(parsed, planned, clips)

        result = pipeline.render_to_file("a person walks", output_name="output")

        pipeline.parser.parse.assert_called_once_with("a person walks")
        pipeline.layout.plan.assert_called_once()
        self.assertIs(pipeline.layout.plan.call_args[0][0], parsed)

        pipeline.motion.generate_for_scene.assert_called_once()
        self.assertIs(pipeline.motion.generate_for_scene.call_args[0][0], planned)

        mock_render.assert_called_once()
        self.assertEqual(result["parsed_scene"], parsed)
        self.assertEqual(result["planned_scene"], planned)
        self.assertEqual(result["motion_clips"], clips)

    def test_empty_prompt_returns_none(self):
        pipeline = Pipeline(PipelineConfig())
        self.assertIsNone(pipeline.render_to_file(""))


class TestPipelineConfig(unittest.TestCase):
    def test_video_path_format(self):
        cfg = PipelineConfig(output_dir="outputs")
        path = cfg.video_path("my_test")
        self.assertIn("my_test.mp4", path)

    def test_duration_propagated_to_planner(self):
        cfg = PipelineConfig(duration=8.0)
        self.assertAlmostEqual(cfg.planner.base_duration, 8.0)


if __name__ == "__main__":
    unittest.main()
