import time

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.dataset.fixture import FIXTURE_SPLIT_SIZES, build_fixture_dataset
from text2motion.dataset.training.captioned import CaptionedMotionDataset
from text2motion.motion.contracts import Split
from text2motion.motion.representation import DIM

CONFIG = MotionDataConfig(mirror_augment=False)


def test_fixture_builds_fast_and_loads_through_the_captioned_dataset(tmp_path):
    started = time.perf_counter()
    splits = build_fixture_dataset(tmp_path)
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    for split, size in FIXTURE_SPLIT_SIZES.items():
        assert len(splits[split]) == size
        dataset = CaptionedMotionDataset(tmp_path, CONFIG, split, seed=2026)
        assert len(dataset) == size
        assert dataset.dropped_clip_ids == []
    item = CaptionedMotionDataset(tmp_path, CONFIG, Split.TRAIN, seed=2026)[0]
    assert item.caption
    assert item.features.shape[-1] == DIM
