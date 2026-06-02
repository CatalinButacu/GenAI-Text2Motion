from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np


class StreamCapabilityError(RuntimeError):
    """Raised when a checkpoint or model config does not support streaming."""



@dataclass(slots=True)
class StreamPacket:
    chunk_id: int
    frames: np.ndarray
    produced_at: float
    consumed_at: float | None = None


@dataclass(slots=True)
class StreamSummary:
    produced_chunks: int
    consumed_chunks: int
    dropped_chunks: int
    bad_chunks: int
    first_chunk_latency_ms: float | None
    inter_chunk_p50_ms: float | None
    inter_chunk_p95_ms: float | None
    wall_time_ms: float


class StreamBus:
    def __init__(self, max_size: int = 32) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self.max_size = max_size
        self.queue: deque[StreamPacket] = deque()
        self.next_chunk_id = 0
        self.dropped_chunks = 0

    def push(self, frames: np.ndarray) -> StreamPacket:
        packet = StreamPacket(
            chunk_id=self.next_chunk_id,
            frames=frames,
            produced_at=time.perf_counter(),
        )
        self.next_chunk_id += 1

        if len(self.queue) >= self.max_size:
            self.queue.popleft()
            self.dropped_chunks += 1

        self.queue.append(packet)
        return packet

    def pop(self) -> StreamPacket | None:
        if not self.queue:
            return None

        packet = self.queue.popleft()
        packet.consumed_at = time.perf_counter()
        return packet

    def size(self) -> int:
        return len(self.queue)


class StreamMetricsCollector:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.first_chunk_latency_ms: float | None = None
        self.produced_chunks = 0
        self.consumed_chunks = 0
        self.bad_chunks = 0
        self.consume_times: list[float] = []

    def observe_produced(self) -> None:
        self.produced_chunks += 1

    def observe_consumed(self, packet: StreamPacket) -> None:
        self.consumed_chunks += 1

        if self.first_chunk_latency_ms is None:
            self.first_chunk_latency_ms = (packet.produced_at - self.started_at) * 1000.0

        if not np.isfinite(packet.frames).all():
            self.bad_chunks += 1

        if packet.consumed_at is not None:
            self.consume_times.append(packet.consumed_at)

    def summary(self, dropped_chunks: int) -> StreamSummary:
        wall_time_ms = (time.perf_counter() - self.started_at) * 1000.0
        intervals_ms = self.consume_intervals_ms()

        return StreamSummary(
            produced_chunks=self.produced_chunks,
            consumed_chunks=self.consumed_chunks,
            dropped_chunks=dropped_chunks,
            bad_chunks=self.bad_chunks,
            first_chunk_latency_ms=self.first_chunk_latency_ms,
            inter_chunk_p50_ms=self.percentile(intervals_ms, 50.0),
            inter_chunk_p95_ms=self.percentile(intervals_ms, 95.0),
            wall_time_ms=wall_time_ms,
        )

    def consume_intervals_ms(self) -> np.ndarray:
        if len(self.consume_times) < 2:
            return np.array([], dtype=np.float64)

        return np.diff(np.array(self.consume_times, dtype=np.float64)) * 1000.0

    @staticmethod
    def percentile(values: np.ndarray, p: float) -> float | None:
        if values.size == 0:
            return None

        return float(np.percentile(values, p))
