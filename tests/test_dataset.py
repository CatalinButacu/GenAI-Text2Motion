"""HumanML3D-263 dataset/dataloader over the regenerated features. Marked `slow` -- needs the
regenerated data + donor texts on disk; skips cleanly when absent."""

from pathlib import Path

import pytest
import torch

from text2motion.data.hml3d.dataset import Hml3dMotionTextDataset, build_dataloader, collate_pad
from text2motion.shared.config import load_config


def _cfg_or_skip():
    cfg = load_config("configs/default.yaml")
    out = cfg.paths.hml3d_out_dir
    if not out or not (Path(out) / "new_joint_vecs").is_dir():
        pytest.skip("regenerated HumanML3D-263 data unavailable")
    if not (Path(out) / "Mean.npy").is_file():
        pytest.skip("Mean/Std unavailable")
    return cfg


@pytest.mark.slow
def test_train_has_mirror_eval_does_not():
    cfg = _cfg_or_skip()
    train = Hml3dMotionTextDataset(cfg.paths, cfg.hml3d, cfg.data, split="train")
    test = Hml3dMotionTextDataset(cfg.paths, cfg.hml3d, cfg.data, split="test")

    assert len(train) > 0 and len(test) > 0
    assert any(i.startswith("M") for i in train._ids)  # mirror augmentation on for train
    assert not any(i.startswith("M") for i in test._ids)  # never mirror the eval set


@pytest.mark.slow
def test_batch_shape_and_normalization():
    cfg = _cfg_or_skip()
    loader = build_dataloader(cfg.paths, cfg.hml3d, cfg.data, split="val", batch_size=4)
    motions, lengths, captions = next(iter(loader))

    assert motions.shape[0] == len(captions) == lengths.shape[0]
    assert motions.shape[2] == cfg.hml3d.dim
    assert motions.shape[1] == int(lengths.max())  # padded to batch max
    assert all(isinstance(c, str) and c for c in captions)

    denorm = loader.dataset.denormalize(motions[0, : lengths[0]])
    assert torch.isfinite(denorm).all()


@pytest.mark.slow
def test_collate_pads_to_max():
    motions = [torch.randn(10, 263), torch.randn(25, 263), torch.randn(7, 263)]
    batch = [(m, m.shape[0], f"caption {i}") for i, m in enumerate(motions)]
    padded, lengths, captions = collate_pad(batch)

    assert padded.shape == (3, 25, 263)
    assert lengths.tolist() == [10, 25, 7]
    assert torch.equal(padded[1], motions[1])  # longest unchanged
    assert torch.count_nonzero(padded[2, 7:]) == 0  # tail zero-padded
