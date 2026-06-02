from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.modules.runtime import (
    StreamBus,
    StreamCapabilityError,
    StreamMetricsCollector,
    StreamPacket,
)
from src.pipeline import Pipeline


class TestStreamBus(unittest.TestCase):

    def test_drop_oldest_when_full(self):
        bus = StreamBus(max_size=2)

        p0 = bus.push(np.zeros((2, 168), dtype=np.float32))
        p1 = bus.push(np.ones((2, 168), dtype=np.float32))
        p2 = bus.push(np.full((2, 168), 2.0, dtype=np.float32))

        self.assertEqual(p0.chunk_id, 0)
        self.assertEqual(p1.chunk_id, 1)
        self.assertEqual(p2.chunk_id, 2)
        self.assertEqual(bus.dropped_chunks, 1)
        self.assertEqual(bus.size(), 2)

        first = bus.pop()
        second = bus.pop()

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(first.chunk_id, 1)
        self.assertEqual(second.chunk_id, 2)

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            StreamBus(max_size=0)


class TestStreamMetricsCollector(unittest.TestCase):

    def test_summary_has_expected_metrics(self):
        metrics = StreamMetricsCollector()

        metrics.observe_produced()
        packet0 = StreamPacket(
            chunk_id=0,
            frames=np.zeros((2, 168), dtype=np.float32),
            produced_at=metrics.started_at + 0.001,
            consumed_at=metrics.started_at + 0.020,
        )
        metrics.observe_consumed(packet0)

        time.sleep(0.005)

        metrics.observe_produced()
        packet1 = StreamPacket(
            chunk_id=1,
            frames=np.zeros((2, 168), dtype=np.float32),
            produced_at=metrics.started_at + 0.030,
            consumed_at=metrics.started_at + 0.040,
        )
        metrics.observe_consumed(packet1)

        summary = metrics.summary(dropped_chunks=3)

        self.assertEqual(summary.produced_chunks, 2)
        self.assertEqual(summary.consumed_chunks, 2)
        self.assertEqual(summary.dropped_chunks, 3)
        self.assertEqual(summary.bad_chunks, 0)
        self.assertIsNotNone(summary.first_chunk_latency_ms)
        self.assertIsNotNone(summary.inter_chunk_p50_ms)
        self.assertIsNotNone(summary.inter_chunk_p95_ms)

    def test_bad_chunk_counts_nan_inf(self):
        metrics = StreamMetricsCollector()

        metrics.observe_produced()
        packet = StreamPacket(
            chunk_id=0,
            frames=np.array([[0.0, np.nan], [np.inf, 1.0]], dtype=np.float32),
            produced_at=metrics.started_at + 0.001,
            consumed_at=metrics.started_at + 0.002,
        )
        metrics.observe_consumed(packet)

        summary = metrics.summary(dropped_chunks=0)
        self.assertEqual(summary.bad_chunks, 1)


CHUNK = np.zeros((15, 168), dtype=np.float32)


class TestStreamBusOverflow(unittest.TestCase):
    """Epic B-3: deterministic overflow / stress path."""

    def test_overflow_drop_count_is_exact(self):
        busSize = 3
        numPushes = 10
        bus = StreamBus(max_size=busSize)

        for i in range(numPushes):
            bus.push(np.full((2, 168), float(i), dtype=np.float32))

        self.assertEqual(bus.dropped_chunks, numPushes - busSize)
        self.assertEqual(bus.size(), busSize)

    def test_after_overflow_oldest_retained_are_newest(self):
        bus = StreamBus(max_size=2)

        for i in range(5):
            bus.push(np.full((1, 168), float(i), dtype=np.float32))

        first = bus.pop()
        second = bus.pop()

        assert first is not None
        assert second is not None
        self.assertEqual(first.chunk_id, 3)
        self.assertEqual(second.chunk_id, 4)


def makePipelineWithMockedMotion(stream_capable: bool = True):
    """Return a Pipeline instance with motion.backend mocked (no checkpoint needed)."""
    pipe = Pipeline.__new__(Pipeline)
    backendMock = MagicMock()

    if stream_capable:
        backendMock.require_stream_capable.return_value = None
    else:
        backendMock.require_stream_capable.side_effect = StreamCapabilityError("bidirectional=True")

    motionMock = MagicMock()
    motionMock.backend = backendMock
    pipe.__dict__["motion"] = motionMock
    return pipe


class TestValidateStreamingContract(unittest.TestCase):
    """Epic A-1: validate_streaming returns the 8 agreed keys and respects max_stream_chunks."""

    EXPECTED_KEYS = {
        "produced_chunks",
        "consumed_chunks",
        "dropped_chunks",
        "bad_chunks",
        "first_chunk_latency_ms",
        "inter_chunk_p50_ms",
        "inter_chunk_p95_ms",
        "wall_time_ms",
    }

    def test_summary_has_exactly_8_keys(self):
        pipe = makePipelineWithMockedMotion()
        chunks = [np.zeros((15, 168), dtype=np.float32) for _ in range(5)]

        with patch.object(pipe, "generate_stream", return_value=iter(chunks)):
            summary = pipe.validate_streaming("any prompt")

        self.assertEqual(set(summary.keys()), self.EXPECTED_KEYS)

    def test_max_stream_chunks_caps_consumed(self):
        pipe = makePipelineWithMockedMotion()
        chunks = [np.zeros((15, 168), dtype=np.float32) for _ in range(20)]

        with patch.object(pipe, "generate_stream", return_value=iter(chunks)):
            summary = pipe.validate_streaming("any prompt", max_stream_chunks=3)

        self.assertEqual(summary["consumed_chunks"], 3)

    def test_wall_time_is_positive(self):
        pipe = makePipelineWithMockedMotion()

        with patch.object(pipe, "generate_stream", return_value=iter([])):
            summary = pipe.validate_streaming("empty prompt")

        wallTime = summary["wall_time_ms"]
        assert wallTime is not None
        self.assertGreater(wallTime, 0.0)

    def test_rejects_zero_max_stream_chunks(self):
        pipe = makePipelineWithMockedMotion()

        with self.assertRaises(ValueError):
            pipe.validate_streaming("a", max_stream_chunks=0)

    def test_rejects_negative_max_stream_chunks(self):
        pipe = makePipelineWithMockedMotion()

        with self.assertRaises(ValueError):
            pipe.validate_streaming("a", max_stream_chunks=-1)


class TestStreamCapabilityGating(unittest.TestCase):
    """Epic D-3: preflight capability check surfaces StreamCapabilityError cleanly."""

    def test_bidirectional_backend_raises_capability_error(self):
        pipe = makePipelineWithMockedMotion(stream_capable=False)

        with self.assertRaises(StreamCapabilityError):
            pipe.validate_streaming("a walk")

    def test_causal_backend_proceeds_without_error(self):
        pipe = makePipelineWithMockedMotion(stream_capable=True)

        with patch.object(pipe, "generate_stream", return_value=iter([])):
            summary = pipe.validate_streaming("a walk")

        self.assertIsNotNone(summary)

    def test_stream_capability_error_is_runtime_error_subtype(self):
        self.assertTrue(issubclass(StreamCapabilityError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
