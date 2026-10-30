import numpy as np
import pytest
import torch

from text2motion.motion.contracts import MotionDataConfig, Split
from text2motion.motion.datasets import (
    CaptionedMotionDataset,
    MotionWindowDataset,
    collate_motion_clips,
)
from text2motion.motion.model import MotionClip
from text2motion.motion.representation import DIM
from text2motion.motion.storage import (
    FEATURES_DIR,
    MEAN_FILE,
    SPLIT_FILES,
    STD_FILE,
    TEXTS_DIR,
    atomic_numpy_save,
)


def __small_dataset_root(root):
    (root / FEATURES_DIR).mkdir()
    (root / TEXTS_DIR).mkdir()
    np.save(root / MEAN_FILE, np.zeros(DIM, np.float32))
    np.save(root / STD_FILE, np.ones(DIM, np.float32))
    for name, frames in (("A", 60), ("B", 80)):
        features = np.random.default_rng(frames).standard_normal((frames, DIM)).astype(np.float32)
        np.save(root / FEATURES_DIR / f"{name}.npy", features)
        (root / TEXTS_DIR / f"{name}.txt").write_text(
            "a person walks#a/DET#0.0#0.0\n", encoding="utf-8"
        )
    (root / SPLIT_FILES[Split.TRAIN]).write_text("A\nB\n", encoding="utf-8")
    return root


def test_a_numpy_save_creates_the_missing_parent_folder(tmp_path):
    target = tmp_path / "new" / "folder" / "array.npy"

    atomic_numpy_save(np.arange(3), target)

    np.testing.assert_array_equal(np.load(target), np.arange(3))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"min_motion_len": 0}, "min_motion_len"),
        ({"min_motion_len": 50, "max_motion_len": 40}, "min_motion_len"),
        ({"sources": ()}, "at least one motion source"),
    ],
)
def test_a_data_config_rejects_impossible_values(values, match):
    with pytest.raises(ValueError, match=match):
        MotionDataConfig(**values)


def test_a_window_dataset_rejects_a_window_below_one(tmp_path):
    root = __small_dataset_root(tmp_path)
    with pytest.raises(ValueError, match="positive frame count"):
        MotionWindowDataset(root, MotionDataConfig(mirror_augment=False), Split.TRAIN, window=0)


def test_a_negative_index_gives_the_same_clip_as_python_indexing(tmp_path):
    dataset = CaptionedMotionDataset(
        __small_dataset_root(tmp_path), MotionDataConfig(mirror_augment=False), Split.TRAIN
    )

    last, by_negative = dataset[len(dataset) - 1], dataset[-1]

    assert last.clip_id == by_negative.clip_id
    torch.testing.assert_close(last.features, by_negative.features)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


@pytest.mark.parametrize(
    "clips",
    [
        [],
        [MotionClip(features=torch.zeros(1, DIM), frame_count=10)],
        [MotionClip(features=torch.zeros(DIM), frame_count=1)],
        [
            MotionClip(features=torch.zeros(4, DIM), frame_count=4),
            MotionClip(features=torch.zeros(4, DIM, dtype=torch.float64), frame_count=4),
        ],
    ],
    ids=["empty", "broadcast_frames", "one_dimension", "mixed_dtype"],
)
def test_collate_rejects_clips_that_would_broadcast_or_mix(clips):
    with pytest.raises(ValueError):
        collate_motion_clips(clips)
