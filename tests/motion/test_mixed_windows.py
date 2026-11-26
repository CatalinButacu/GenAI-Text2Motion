import numpy as np
import pytest

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.dataset.training.windows import MixedWindowDataset, MotionWindowDataset
from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MEAN_FILE, STD_FILE
from text2motion.motion.storage import FEATURES_DIR, SPLIT_FILES

DIM = 263
WINDOW = 70


def write_root(root, names_and_frames, with_split):
    (root / FEATURES_DIR).mkdir(parents=True)
    np.save(root / MEAN_FILE, np.zeros(DIM, np.float32))
    np.save(root / STD_FILE, np.ones(DIM, np.float32))
    for name, frames in names_and_frames:
        features = np.random.default_rng(frames).standard_normal((frames, DIM)).astype(np.float32)
        np.save(root / FEATURES_DIR / f"{name}.npy", features)
    if with_split:
        (root / SPLIT_FILES[Split.TRAIN]).write_text(
            "\n".join(name for name, _ in names_and_frames) + "\n", encoding="utf-8"
        )
    return root


def build_pair(tmp_path, extension_count):
    primary_root = write_root(tmp_path / "hml", [("A", 80), ("B", 90)], with_split=True)
    extension_root = write_root(
        tmp_path / "amass",
        [(f"X{index}", 60 + index) for index in range(extension_count)],
        with_split=False,
    )
    data = MotionDataConfig(mirror_augment=False)
    primary = MotionWindowDataset(primary_root, data, Split.TRAIN, WINDOW, seed=2026)
    extension = MotionWindowDataset(
        extension_root,
        data,
        Split.TRAIN,
        WINDOW,
        seed=2026,
        names=sorted(f"X{index}" for index in range(extension_count)),
        scaler=primary.scaler,
    )
    return primary, extension


def test_an_even_share_adds_one_extension_clip_per_primary_clip(tmp_path):
    primary, extension = build_pair(tmp_path, extension_count=5)
    mixed = MixedWindowDataset(primary, extension, 0.5, seed=2026)

    assert mixed.extension_count == len(primary)
    assert len(mixed) == 2 * len(primary)


def test_a_zero_share_keeps_the_primary_clips_only(tmp_path):
    primary, extension = build_pair(tmp_path, extension_count=5)
    mixed = MixedWindowDataset(primary, extension, 0.0, seed=2026)

    assert mixed.extension_count == 0
    assert len(mixed) == len(primary)


def test_a_share_of_one_or_more_is_refused(tmp_path):
    primary, extension = build_pair(tmp_path, extension_count=5)
    with pytest.raises(ValueError, match="extension share"):
        MixedWindowDataset(primary, extension, 1.0, seed=2026)


def test_the_plan_of_an_epoch_is_repeatable_and_changes_with_the_epoch(tmp_path):
    primary, extension = build_pair(tmp_path, extension_count=20)
    mixed = MixedWindowDataset(primary, extension, 0.5, seed=2026)

    mixed.set_epoch(3)
    first = list(mixed._plan)
    mixed.set_epoch(3)
    assert list(mixed._plan) == first

    mixed.set_epoch(4)
    assert list(mixed._plan) != first


def test_a_mixed_item_returns_a_clip_of_its_own_source(tmp_path):
    primary, extension = build_pair(tmp_path, extension_count=5)
    mixed = MixedWindowDataset(primary, extension, 0.5, seed=2026)

    clip_ids = {mixed[index].clip_id or "" for index in range(len(mixed))}

    assert {"A", "B"} <= clip_ids
    assert any(clip_id.startswith("X") for clip_id in clip_ids)
