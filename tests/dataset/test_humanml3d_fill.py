import json
from pathlib import Path

import numpy as np

from text2motion.dataset.sources.humanml3d.source import HumanML3DSource


def make_official(folder: Path) -> None:
    (folder / "new_joint_vecs").mkdir(parents=True)
    (folder / "index.csv").write_text(
        "source_path,start_frame,end_frame,new_name\n"
        "./pose_data/A/a.npy,0,10,000001.npy\n"
        "./pose_data/B/b.npy,0,10,000002.npy\n",
        encoding="utf-8",
    )
    (folder / "train.txt").write_text("000001\n", encoding="utf-8")
    (folder / "val.txt").write_text("000002\n", encoding="utf-8")
    (folder / "test.txt").write_text("", encoding="utf-8")
    rng = np.random.default_rng(2026)
    for name in ("000001", "M000001", "000002", "M000002"):
        np.save(
            folder / "new_joint_vecs" / f"{name}.npy", rng.normal(size=(8, 263)).astype(np.float32)
        )


def test_fill_writes_training_clips_and_their_mirror_only(tmp_path: Path) -> None:
    official = tmp_path / "official"
    make_official(official)
    out = tmp_path / "out"
    joints = out / "joints"
    joints.mkdir(parents=True)
    (out / "index_manifest.json").write_text(
        json.dumps({"missing_names": ["000001.npy", "000002.npy"]}), encoding="utf-8"
    )
    filled = HumanML3DSource(official).fill_missing(out, joints, fps=60)
    assert filled == 1
    assert (joints / "000001.npy").is_file()
    assert (joints / "M000001.npy").is_file()
    assert not (joints / "000002.npy").exists()
    assert not (joints / "M000002.npy").exists()


def test_filled_clips_have_the_requested_frame_rate(tmp_path: Path) -> None:
    official = tmp_path / "official"
    make_official(official)
    out = tmp_path / "out"
    joints = out / "joints"
    joints.mkdir(parents=True)
    (out / "index_manifest.json").write_text(
        json.dumps({"missing_names": ["000001.npy"]}), encoding="utf-8"
    )
    HumanML3DSource(official).fill_missing(out, joints, fps=60)
    clip = np.load(joints / "000001.npy")
    assert clip.shape[0] == 8 * 3 - 2
    assert clip.shape[1:] == (22, 3)


def test_evaluation_splits_are_filled_when_asked(tmp_path: Path) -> None:
    official = tmp_path / "official"
    make_official(official)
    out = tmp_path / "out"
    joints = out / "joints"
    joints.mkdir(parents=True)
    (out / "index_manifest.json").write_text(
        json.dumps({"missing_names": ["000001.npy", "000002.npy"]}), encoding="utf-8"
    )
    filled = HumanML3DSource(official).fill_missing(
        out, joints, fps=60, splits=("train", "val", "test")
    )
    assert filled == 2
    assert (joints / "000002.npy").is_file()
