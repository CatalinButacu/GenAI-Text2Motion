import numpy as np
import pytest

from text2motion.motion.contracts import MotionDataConfig
from text2motion.motion.datasets import CaptionedMotionDataset, MotionWindowDataset

CAPTION = "a person walks#a/DET#0.0#0.0"


def __dataset_root(tmp_path, frames=(80, 80, 20)):
    (tmp_path / "new_joint_vecs").mkdir()
    (tmp_path / "texts").mkdir()
    for index, count in enumerate(frames):
        name = f"{index:06d}"
        np.save(tmp_path / "new_joint_vecs" / f"{name}.npy", np.zeros((count, 263), np.float32))
        (tmp_path / "texts" / f"{name}.txt").write_text(CAPTION, encoding="utf-8")
    np.save(tmp_path / "Mean.npy", np.zeros(263, np.float32))
    np.save(tmp_path / "Std.npy", np.ones(263, np.float32))
    names = "\n".join(f"{index:06d}" for index in range(len(frames)))
    (tmp_path / "train.txt").write_text(names, encoding="utf-8")
    return tmp_path


DATA = MotionDataConfig(mirror_augment=False, min_motion_len=40, max_motion_len=64)


def test_short_clips_are_still_dropped_quietly(tmp_path):
    dataset = CaptionedMotionDataset(__dataset_root(tmp_path), DATA, "train")

    assert dataset.clip_ids == ["000000", "000001"]


def test_a_missing_feature_file_stops_the_captioned_loader(tmp_path):
    root = __dataset_root(tmp_path)
    (root / "new_joint_vecs" / "000001.npy").unlink()

    with pytest.raises(FileNotFoundError, match="no feature file.*000001"):
        CaptionedMotionDataset(root, DATA, "train")


def test_a_missing_caption_file_stops_the_captioned_loader(tmp_path):
    root = __dataset_root(tmp_path)
    (root / "texts" / "000000.txt").unlink()

    with pytest.raises(FileNotFoundError, match="no caption file.*000000"):
        CaptionedMotionDataset(root, DATA, "train")


def test_a_missing_feature_file_stops_the_window_loader(tmp_path):
    root = __dataset_root(tmp_path)
    (root / "new_joint_vecs" / "000000.npy").unlink()

    with pytest.raises(FileNotFoundError, match="no feature file.*000000"):
        MotionWindowDataset(root, DATA, "train", 32)
