import numpy as np
import pandas as pd
import pytest

from text2motion.motion.preparation import (
    AMASS_DATASET_RENAME,
    DATASET_HEAD_TRIM_S,
    dataset_of,
    head_trim_frames,
    heldout_sources,
    resolve_pose_path,
)

FPS = 20


def test_head_trim_agrees_for_official_and_renamed_folders():
    for official, short in AMASS_DATASET_RENAME.items():
        assert dataset_of(f"./pose_data/{official}/sub/clip_poses.npz") == official
        assert dataset_of(f"{short}/sub/clip.npy") == official
        assert head_trim_frames(f"./pose_data/{official}/s/c_poses.npz", FPS) == head_trim_frames(
            f"{short}/s/c.npy", FPS
        )


def test_every_trimmed_dataset_is_reachable_from_its_renamed_folder():
    for official, seconds in DATASET_HEAD_TRIM_S.items():
        short = AMASS_DATASET_RENAME.get(official, official)
        assert head_trim_frames(f"{short}/s/c.npy", FPS) == int(round(seconds * FPS)), (
            f"{official} loses its head trim when reached via folder {short!r}"
        )


def test_untrimmed_dataset_gets_zero():
    assert head_trim_frames("CMU/01/01_01.npy", FPS) == 0


def build_corpus_tree(tmp_path, spelling: str):
    pose_root = tmp_path / "pose_data"
    (pose_root / "CMU" / "01").mkdir(parents=True)
    np.save(pose_root / "CMU" / "01" / spelling, np.zeros((4, 22, 3), np.float32))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "val.txt").write_text("000001\n", encoding="utf-8")
    (out_dir / "test.txt").write_text("M000002\n", encoding="utf-8")

    index = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "source_path": ["./pose_data/CMU/01/01_01_poses.npz"],
            "new_name": ["000001.npy"],
        }
    ).to_csv(index, index=False)
    return index, out_dir, pose_root


def test_heldout_clip_is_excluded_when_its_capture_resolves(tmp_path):
    index, out_dir, pose_root = build_corpus_tree(tmp_path, "01_01_poses.npy")

    excluded = heldout_sources(index, out_dir, pose_root)

    assert excluded == {(pose_root / "CMU" / "01" / "01_01_poses.npy").resolve()}


def test_guard_fails_loud_when_the_folder_exists_but_spelling_is_unknown(tmp_path):
    index, out_dir, pose_root = build_corpus_tree(tmp_path, "01_01_UNEXPECTED.npy")

    with pytest.raises(RuntimeError, match="leakage guard failed"):
        heldout_sources(index, out_dir, pose_root)


def test_guard_stays_quiet_when_the_capture_is_simply_not_downloaded(tmp_path):
    index, out_dir, pose_root = build_corpus_tree(tmp_path, "01_01_poses.npy")
    index2 = tmp_path / "index2.csv"
    pd.DataFrame(
        {
            "source_path": ["./pose_data/ACCAD/99/absent_poses.npz"],
            "new_name": ["000001.npy"],
        }
    ).to_csv(index2, index=False)

    assert heldout_sources(index2, out_dir, pose_root) == set()


def test_resolver_accepts_the_known_spellings(tmp_path):
    pose_root = tmp_path / "pose_data"
    (pose_root / "CMU" / "01").mkdir(parents=True)
    np.save(pose_root / "CMU" / "01" / "01_01_stageii.npy", np.zeros((2, 22, 3), np.float32))

    found = resolve_pose_path("./pose_data/CMU/01/01_01_poses.npz", pose_root)

    assert found is not None and found.name == "01_01_stageii.npy"


def test_resolver_matches_underscored_smplx_release_names(tmp_path):
    pose_root = tmp_path / "pose_data"
    folder = pose_root / "ACCAD" / "Male1General_c3d"
    folder.mkdir(parents=True)
    np.save(folder / "General_A9_-___Lie_(forward)_stageii.npy", np.zeros((2, 22, 3), np.float32))

    found = resolve_pose_path(
        "./pose_data/ACCAD/Male1General_c3d/General A9 -   Lie (forward)_poses.npy", pose_root
    )

    assert found is not None, "spaced index name must match the underscored SMPL-X file"
    assert found.name == "General_A9_-___Lie_(forward)_stageii.npy"


def test_heldout_guard_excludes_underscored_captures(tmp_path):
    pose_root = tmp_path / "pose_data"
    folder = pose_root / "ACCAD" / "Female1Walking_c3d"
    folder.mkdir(parents=True)
    target = folder / "B5_-_walk_backwards_stageii.npy"
    np.save(target, np.zeros((2, 22, 3), np.float32))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "val.txt").write_text("000001\n", encoding="utf-8")
    (out_dir / "test.txt").write_text("", encoding="utf-8")

    index = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "source_path": ["./pose_data/ACCAD/Female1Walking_c3d/B5 - walk backwards_poses.npy"],
            "new_name": ["000001.npy"],
        }
    ).to_csv(index, index=False)

    assert heldout_sources(index, out_dir, pose_root) == {target.resolve()}
