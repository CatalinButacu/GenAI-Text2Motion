"""Streaming performance proof tests (US-10).

These tests verify the constant-memory streaming claim from the thesis.
Marked @pytest.mark.slow — skip with: pytest -m "not slow"
"""
from __future__ import annotations

import pytest

from tests.benchmarks.benchmark_streaming import (
    detectMemorySlope,
    streamingPeakKB,
    sweepSequenceLengths,
)

# Streaming peak memory must not grow faster than this many KB per additional frame.
# At 168-dim float32 chunks, a truly constant-memory path should be well below 1 KB/frame.
STREAM_SLOPE_THRESHOLD_KB_PER_FRAME = 1.0

# Naive baseline must grow faster than streaming (by at least 2x slope ratio).
MIN_NAIVE_VS_STREAM_SLOPE_RATIO = 2.0


@pytest.mark.slow
class TestStreamingMemoryConstancy:
    def test_streaming_slope_near_zero(self):
        """Streaming peak KB must not grow linearly with T."""
        results = sweepSequenceLengths(lengths=[100, 500, 1000, 2000])
        slopes = detectMemorySlope(results)
        assert slopes["stream_slope"] < STREAM_SLOPE_THRESHOLD_KB_PER_FRAME, (
            f"Streaming memory slope {slopes['stream_slope']:.4f} KB/frame exceeds threshold "
            f"{STREAM_SLOPE_THRESHOLD_KB_PER_FRAME} KB/frame — constant-memory claim violated"
        )

    @pytest.mark.slow
    def test_naive_slope_exceeds_stream_by_factor(self):
        """Naive path must use significantly more memory per frame than streaming."""
        results = sweepSequenceLengths(lengths=[100, 500, 1000, 2000])
        slopes = detectMemorySlope(results)

        if slopes["stream_slope"] > 0:
            ratio = slopes["naive_slope"] / slopes["stream_slope"]
            assert ratio >= MIN_NAIVE_VS_STREAM_SLOPE_RATIO, (
                f"Naive slope ({slopes['naive_slope']:.4f}) vs stream slope "
                f"({slopes['stream_slope']:.4f}) ratio={ratio:.2f} < "
                f"required {MIN_NAIVE_VS_STREAM_SLOPE_RATIO}"
            )

    def test_streaming_peak_smaller_than_naive_at_T4000(self):
        """At T=4000, streaming peak must be < naive peak."""
        from tests.benchmarks.benchmark_streaming import naivePeakKB

        streamKB = streamingPeakKB(4000)
        nKB = naivePeakKB(4000)
        assert streamKB < nKB, (
            f"Streaming ({streamKB:.1f} KB) >= naive ({nKB:.1f} KB) at T=4000"
        )
