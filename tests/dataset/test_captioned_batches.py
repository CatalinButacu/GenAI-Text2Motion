import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from text2motion.dataset.contracts import ClipWeighting, MotionDataConfig
from text2motion.dataset.fixture import build_fixture_dataset
from text2motion.dataset.training.batching import collate_motion_clips
from text2motion.dataset.training.captioned import CaptionedMotionDataset
from text2motion.motion.contracts import Split

RECORD = Path(__file__).with_name("captioned_batches.json")
SEED = 2026
EPOCHS = (0, 1)


def batch_record(batch) -> dict:
    return {
        "captions": list(batch.captions),
        "lengths": batch.lengths.tolist(),
        "shape": list(batch.features.shape),
        "sha256": hashlib.sha256(batch.features.numpy().tobytes()).hexdigest(),
    }


def split_records(root: Path, split: Split) -> dict:
    config = MotionDataConfig(
        mirror_augment=False,
        min_motion_len=40,
        max_motion_len=48,
        clip_weighting=ClipWeighting.UNIFORM_REPLACEMENT,
    )
    dataset = CaptionedMotionDataset(root, config, split, seed=SEED)
    torch.manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_motion_clips)
    weights = dataset.clip_weights
    epochs = []

    for epoch in EPOCHS:
        dataset.set_epoch(epoch)
        epochs.append([batch_record(batch) for batch in loader])

    return {
        "clip_ids": dataset.clip_ids,
        "captions": dataset.captions(),
        "clip_weights": None if weights is None else weights.tolist(),
        "epochs": epochs,
    }


@pytest.mark.parametrize("split", list(Split))
def test_seeded_batches_equal_the_values_recorded_before_the_split(tmp_path, split):
    build_fixture_dataset(tmp_path)
    recorded = json.loads(RECORD.read_text(encoding="utf-8"))

    assert split_records(tmp_path, split) == recorded[split.value]
