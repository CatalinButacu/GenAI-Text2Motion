from pathlib import Path

import numpy as np

from text2motion.dataset.sources.amass.poses import pose_path_for, same_pose


def test_pose_file_keeps_the_relative_path_and_uses_npy(tmp_path: Path) -> None:
    amass_root = tmp_path / "amass"
    pose_root = tmp_path / "pose_data"
    src = amass_root / "KIT" / "4" / "WalkInCounterClockwiseCircle02_stageii.npz"
    assert pose_path_for(src, amass_root, pose_root) == (
        pose_root / "KIT" / "4" / "WalkInCounterClockwiseCircle02_stageii.npy"
    )


def test_same_pose_accepts_values_within_tolerance() -> None:
    saved = np.zeros((3, 22, 3), dtype=np.float32)
    assert same_pose(saved + 1e-6, saved)


def test_same_pose_rejects_a_missing_or_different_shape() -> None:
    saved = np.zeros((3, 22, 3), dtype=np.float32)
    assert not same_pose(None, saved)
    assert not same_pose(np.zeros((4, 22, 3), dtype=np.float32), saved)


def test_same_pose_rejects_values_outside_tolerance() -> None:
    saved = np.zeros((3, 22, 3), dtype=np.float32)
    assert not same_pose(saved + 1.0, saved)
