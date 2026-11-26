from pathlib import Path

import numpy as np
import pytest

from text2motion.dataset.sources.humanml3d.source import reference_joints_path
from text2motion.motion.contracts import PreparationRequest


def test_default_reference_clip_is_the_official_one(tmp_path: Path) -> None:
    request = PreparationRequest(out_dir=tmp_path)
    assert request.reference_clip == "000021"


def test_lookup_uses_the_requested_clip(tmp_path: Path) -> None:
    np.save(tmp_path / "012314.npy", np.zeros((3, 22, 3), dtype=np.float32))
    assert reference_joints_path(tmp_path, "012314") == tmp_path / "012314.npy"


def test_missing_reference_names_the_requested_clip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="012314"):
        reference_joints_path(tmp_path, "012314")
