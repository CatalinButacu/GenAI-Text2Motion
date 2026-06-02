"""E2E latency benchmark harness (US-11).

Times each pipeline stage individually using time.perf_counter().
Requires a trained checkpoint; skip otherwise.

Run directly:
    python tests/benchmarks/benchmark_e2e.py

Or via pytest (slow + checkpoint guard):
    pytest tests/test_e2e_benchmark.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_PATH = "checkpoints/motion_ssm/best_model.pt"
TEST_PROMPTS = [
    "a person walks forward",
    "a person sits down",
    "a person waves their hand",
]


@dataclass
class StageTimings:
    prompt: str
    parse_ms: float
    plan_ms: float
    motion_ms: float
    total_ms: float


def runE2EBenchmark(
    prompts: list[str] | None = None,
    device: str = "cpu",
) -> list[StageTimings]:
    """Run the full pipeline and record per-stage wall-clock time.

    Parameters
    ----------
    prompts:
        List of text prompts to benchmark. Defaults to TEST_PROMPTS.
    device:
        Device string ("cpu" or "cuda").

    Returns
    -------
    List of StageTimings, one per prompt.
    """
    from src.modules.planner.config import PlannerConfig
    from src.modules.understanding.config import ParserConfig
    from src.pipeline import Pipeline
    from src.shared.config import MotionConfig, PipelineConfig

    if prompts is None:
        prompts = TEST_PROMPTS

    cfg = PipelineConfig(
        device=device,
        duration=3.0,
        fps=24,
        understanding=ParserConfig(),
        planner=PlannerConfig(),
        motion=MotionConfig(),
    )
    pipeline = Pipeline(cfg)
    timings: list[StageTimings] = []

    for prompt in prompts:
        # M1: parse
        t0 = time.perf_counter()
        parsedScene = pipeline.parser.parse(prompt)
        parseMs = (time.perf_counter() - t0) * 1000.0

        # M2: plan
        t0 = time.perf_counter()
        plannedScene = pipeline.layout.plan(parsedScene)
        planMs = (time.perf_counter() - t0) * 1000.0

        # M4: motion
        t0 = time.perf_counter()
        pipeline.motion.generate_for_scene(plannedScene)
        motionMs = (time.perf_counter() - t0) * 1000.0

        totalMs = parseMs + planMs + motionMs
        timings.append(
            StageTimings(
                prompt=prompt,
                parse_ms=parseMs,
                plan_ms=planMs,
                motion_ms=motionMs,
                total_ms=totalMs,
            )
        )

    return timings


def printE2ETable(timings: list[StageTimings]) -> None:
    print(f"\n{'Prompt':<40} {'Parse':>8} {'Plan':>8} {'Motion':>10} {'Total':>10}")
    print("-" * 80)

    for t in timings:
        truncPrompt = t.prompt[:38] + ".." if len(t.prompt) > 40 else t.prompt
        print(
            f"{truncPrompt:<40} {t.parse_ms:>7.1f}ms {t.plan_ms:>7.1f}ms "
            f"{t.motion_ms:>9.1f}ms {t.total_ms:>9.1f}ms"
        )

    print()


if __name__ == "__main__":
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Checkpoint not found at {CHECKPOINT_PATH} — skipping E2E benchmark")
    else:
        timings = runE2EBenchmark()
        printE2ETable(timings)
