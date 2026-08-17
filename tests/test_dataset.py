from pathlib import Path

import pytest
import torch

from text2motion.app.config import load_config
from text2motion.motion.dataset import (
    FEATURES_DIR,
    MEAN_FILE,
    MotionRepository,
    Split,
    collate_clips,
)
from text2motion.motion.model import MotionClip
from text2motion.motion.representation import DIM


def _repository_or_skip() -> MotionRepository:
    cfg = load_config("configs/default.yaml")
    out = cfg.paths.hml3d_out_dir
    if not out or not (Path(out) / FEATURES_DIR).is_dir():
        pytest.skip("regenerated HumanML3D-263 data unavailable")
    if not (Path(out) / MEAN_FILE).is_file():
        pytest.skip("Mean/Std unavailable")
    return MotionRepository(
        root=Path(out), data=cfg.data, texts_dir=cfg.paths.texts_dir, seed=cfg.seed
    )


@pytest.mark.slow
def test_train_has_mirror_eval_does_not():
    repository = _repository_or_skip()
    train = repository.split(Split.TRAIN)
    test = repository.split(Split.TEST)

    assert len(train) > 0 and len(test) > 0
    assert any(clip_id.startswith("M") for clip_id in train.clip_ids)
    assert not any(clip_id.startswith("M") for clip_id in test.clip_ids)


@pytest.mark.slow
def test_batch_shape_and_normalization():
    repository = _repository_or_skip()
    loader = repository.loader(Split.VALIDATION, batch_size=4)
    batch = next(iter(loader))

    assert batch.features.shape[0] == len(batch.captions) == batch.lengths.shape[0]
    assert batch.features.shape[2] == DIM
    assert batch.features.shape[1] == int(batch.lengths.max())
    assert all(isinstance(caption, str) and caption for caption in batch.captions)

    denorm = repository.scaler().denormalize(batch.features[0, : batch.lengths[0]])
    assert torch.isfinite(denorm).all()


def test_collate_pads_to_max():
    motions = [torch.randn(10, DIM), torch.randn(25, DIM), torch.randn(7, DIM)]
    clips = [
        MotionClip(features=motion, frame_count=motion.shape[0], caption=f"caption {index}")
        for index, motion in enumerate(motions)
    ]
    batch = collate_clips(clips)

    assert batch.features.shape == (3, 25, DIM)
    assert batch.lengths.tolist() == [10, 25, 7]
    assert torch.equal(batch.features[1], motions[1])
    assert torch.count_nonzero(batch.features[2, 7:]) == 0
