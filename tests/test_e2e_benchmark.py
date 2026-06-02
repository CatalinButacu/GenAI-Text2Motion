"""E2E latency benchmark tests (US-11).

Marked @pytest.mark.slow and guarded with skipif when checkpoint is missing.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.benchmarks.benchmark_e2e import CHECKPOINT_PATH, StageTimings, runE2EBenchmark

CHECKPOINT_PRESENT = Path(CHECKPOINT_PATH).exists()


class TestStageTimings:
    def test_dataclass_fields(self):
        t = StageTimings(
            prompt="walk",
            parse_ms=1.0,
            plan_ms=2.0,
            motion_ms=100.0,
            total_ms=103.0,
        )
        assert t.total_ms == pytest.approx(103.0)

    def test_all_timings_positive(self):
        t = StageTimings(
            prompt="test",
            parse_ms=0.5,
            plan_ms=0.5,
            motion_ms=10.0,
            total_ms=11.0,
        )
        assert t.parse_ms >= 0.0
        assert t.plan_ms >= 0.0
        assert t.motion_ms >= 0.0
        assert t.total_ms >= 0.0


@pytest.mark.slow
@pytest.mark.skipif(not CHECKPOINT_PRESENT, reason="Checkpoint not available")
class TestRunE2EBenchmark:
    def test_returns_one_result_per_prompt(self):
        prompts = ["a person walks", "a person sits"]
        timings = runE2EBenchmark(prompts=prompts, device="cpu")
        assert len(timings) == len(prompts)

    def test_all_timings_positive(self):
        timings = runE2EBenchmark(prompts=["a person walks"], device="cpu")
        t = timings[0]
        assert t.parse_ms >= 0.0
        assert t.plan_ms >= 0.0
        assert t.motion_ms >= 0.0

    def test_total_equals_sum_of_stages(self):
        timings = runE2EBenchmark(prompts=["a person walks"], device="cpu")
        t = timings[0]
        assert t.total_ms == pytest.approx(t.parse_ms + t.plan_ms + t.motion_ms, abs=1.0)

    def test_motion_stage_dominates(self):
        """Motion generation should take longer than parsing."""
        timings = runE2EBenchmark(prompts=["a person walks forward"], device="cpu")
        t = timings[0]
        assert t.motion_ms > t.parse_ms
