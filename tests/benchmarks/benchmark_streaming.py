"""Streaming memory / throughput benchmark (US-10).

Sweeps over increasing sequence lengths T and measures peak RSS memory via
tracemalloc.  The streaming path should exhibit O(1) memory growth (constant
wrt T) while a naive cumulative-buffer baseline grows O(T).

Run directly:
    python tests/benchmarks/benchmark_streaming.py

Or via pytest (slow mark):
    pytest tests/test_streaming_perf.py
"""
from __future__ import annotations

import tracemalloc
from dataclasses import dataclass

import numpy as np

from src.modules.runtime.stream_runtime import StreamBus

# Each synthetic chunk is one frame of 168-dim SMPL-X pose (float32 ~ 672 bytes).
POSE_DIM = 168
CHUNK_FRAMES = 1
CHUNK_BYTES = POSE_DIM * CHUNK_FRAMES * 4  # float32


@dataclass
class BenchResult:
    seq_len: int
    stream_peak_kb: float
    naive_peak_kb: float
    mem_ratio: float  # naive / stream; should approach T/constant


def streamingPeakKB(seqLen: int) -> float:
    """Measure peak RSS for push+pop cycle over *seqLen* frames via StreamBus."""
    bus = StreamBus(max_size=32)
    tracemalloc.start()
    chunk = np.zeros((CHUNK_FRAMES, POSE_DIM), dtype=np.float32)

    for _ in range(seqLen):
        bus.push(chunk)
        bus.pop()

    _, peakBytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peakBytes / 1024.0


def naivePeakKB(seqLen: int) -> float:
    """Measure peak RSS for naive accumulate-all approach over *seqLen* frames."""
    tracemalloc.start()
    buffer: list[np.ndarray] = []
    chunk = np.zeros((CHUNK_FRAMES, POSE_DIM), dtype=np.float32)

    for _ in range(seqLen):
        buffer.append(chunk.copy())

    _ = np.concatenate(buffer, axis=0)
    _, peakBytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peakBytes / 1024.0


def sweepSequenceLengths(
    lengths: list[int] | None = None,
) -> list[BenchResult]:
    """Run the streaming vs naive memory benchmark at each sequence length."""
    if lengths is None:
        lengths = [100, 500, 1000, 2000, 4000]

    results: list[BenchResult] = []

    for seqLen in lengths:
        streamKB = streamingPeakKB(seqLen)
        naiveKB = naivePeakKB(seqLen)
        ratio = naiveKB / streamKB if streamKB > 0 else float("inf")
        results.append(
            BenchResult(
                seq_len=seqLen,
                stream_peak_kb=streamKB,
                naive_peak_kb=naiveKB,
                mem_ratio=ratio,
            )
        )

    return results


def detectMemorySlope(results: list[BenchResult]) -> dict[str, float]:
    """Fit a linear slope to streaming peak KB vs sequence length.

    A constant-memory streaming path should have slope ≈ 0 (flat line).
    The naive path should show a steep positive slope.

    Returns
    -------
    dict with 'stream_slope' and 'naive_slope' in KB/frame units.
    """
    lengths = np.array([r.seq_len for r in results], dtype=float)
    streamKBs = np.array([r.stream_peak_kb for r in results], dtype=float)
    naiveKBs = np.array([r.naive_peak_kb for r in results], dtype=float)
    streamFit = np.polyfit(lengths, streamKBs, 1)
    naiveFit = np.polyfit(lengths, naiveKBs, 1)
    return {"stream_slope": float(streamFit[0]), "naive_slope": float(naiveFit[0])}


def printBenchTable(results: list[BenchResult]) -> None:
    print(f"\n{'T':>6} {'Stream KB':>12} {'Naive KB':>12} {'Ratio':>8}")
    print("-" * 44)

    for r in results:
        print(
            f"{r.seq_len:>6} {r.stream_peak_kb:>12.1f} {r.naive_peak_kb:>12.1f} "
            f"{r.mem_ratio:>8.2f}x"
        )

    print()


if __name__ == "__main__":
    results = sweepSequenceLengths()
    printBenchTable(results)
    slopes = detectMemorySlope(results)
    print(f"Streaming memory slope : {slopes['stream_slope']:.4f} KB/frame")
    print(f"Naive memory slope     : {slopes['naive_slope']:.4f} KB/frame")
