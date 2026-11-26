from pathlib import Path

import numpy as np
import pytest

from text2motion.dataset.sources.humanml3d import build
from text2motion.dataset.sources.humanml3d.build import (
    Hml3dPreparationLayout,
    encode_joint_clips_to_hml3d263,
)
from text2motion.motion.contracts import PreparationRequest


def build_run(out: Path) -> tuple[PreparationRequest, Hml3dPreparationLayout]:
    layout = Hml3dPreparationLayout(out)
    layout.mkdirs()
    rng = np.random.default_rng(2026)
    np.save(layout.joints / "000021.npy", rng.normal(size=(30, 22, 3)).astype(np.float32))
    np.save(layout.joints / "000001.npy", rng.normal(size=(30, 22, 3)).astype(np.float32))
    request = PreparationRequest(out_dir=out, fps=20, joint_count=22, reference_clip="000021")
    return request, layout


def test_data_error_on_one_clip_is_recorded_then_run_fails(tmp_path: Path, monkeypatch) -> None:
    request, layout = build_run(tmp_path)

    def bad_values(source, params, joint_count):
        raise ValueError("non-finite features")

    monkeypatch.setattr(build, "joints_to_263", bad_values)
    with pytest.raises(RuntimeError, match="failed"):
        encode_joint_clips_to_hml3d263(request, layout)


def test_programming_error_stops_the_run(tmp_path: Path, monkeypatch) -> None:
    request, layout = build_run(tmp_path)

    def broken(source, params, joint_count):
        raise TypeError("bug in our code")

    monkeypatch.setattr(build, "joints_to_263", broken)
    with pytest.raises(TypeError, match="bug in our code"):
        encode_joint_clips_to_hml3d263(request, layout)
