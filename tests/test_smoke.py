"""End-to-end smoke test: main.py must produce a non-empty MP4."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_pipeline_runs(tmpPath: Path) -> None:
    ckpt = ROOT / "checkpoints" / "motion_ssm" / "best_model.pt"
    rvq = ROOT / "checkpoints" / "rvq_tokenizer" / "best_model.pt"

    if not ckpt.exists() or not rvq.exists():
        pytest.skip(f"missing trained checkpoints ({ckpt}, {rvq})")

    out_dir = tmpPath / "outputs"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "a person walks forward",
        "--name", "smoke",
        "--output-dir", str(out_dir),
        "--duration", "2",
        "--fps", "30",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        pytest.fail(f"main.py exit={proc.returncode}")

    video = out_dir / "videos" / "smoke.mp4"
    assert video.exists(), f"expected video at {video}"
    assert video.stat().st_size > 10_000, f"video too small: {video.stat().st_size} bytes"


def test_imports_resolve() -> None:
    """Fast sanity check: pipeline composition still imports cleanly."""
    from src.pipeline import Pipeline  # noqa: F401
    from src.shared.config import PipelineConfig

    cfg = PipelineConfig(duration=5.0, fps=30)
    assert cfg.render.fps == 30
    assert cfg.planner.baseDuration == 5.0
