import numpy as np
import pytest
from torch.utils.data import DataLoader

from text2motion.motion.contracts import MotionDataConfig
from text2motion.motion.datasets import (
    CaptionedMotionDataset,
    MotionWindowDataset,
    collate_motion_clips,
)

CLIPS = 6


def __dataset_root(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "new_joint_vecs").mkdir()
    (tmp_path / "texts").mkdir()
    for index in range(CLIPS):
        name = f"{index:06d}"
        features = rng.normal(size=(200, 263)).astype(np.float32)
        np.save(tmp_path / "new_joint_vecs" / f"{name}.npy", features)
        captions = [f"caption {name} {variant}#a/DET#0.0#0.0" for variant in range(8)]
        (tmp_path / "texts" / f"{name}.txt").write_text("\n".join(captions), encoding="utf-8")
    np.save(tmp_path / "Mean.npy", np.zeros(263, np.float32))
    np.save(tmp_path / "Std.npy", np.ones(263, np.float32))
    names = "\n".join(f"{index:06d}" for index in range(CLIPS))
    (tmp_path / "train.txt").write_text(names, encoding="utf-8")
    return tmp_path


def __datasets(root):
    data = MotionDataConfig(mirror_augment=False, min_motion_len=40, max_motion_len=64)
    return {
        "window": MotionWindowDataset(root, data, "train", 32, seed=1),
        "captioned": CaptionedMotionDataset(root, data, "train", seed=1),
    }


def __epoch_draws(
    dataset: MotionWindowDataset | CaptionedMotionDataset, workers: DataLoader, epoch: int
) -> list[object]:
    dataset.set_epoch(epoch)
    draws: list[object] = []
    for batch in workers:
        draws.extend(float(value) for value in batch.features[:, 0, 0])
        draws.extend(batch.captions or [])
    return draws


def __loader(dataset, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_motion_clips,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
    )


@pytest.mark.parametrize("kind", ["window", "captioned"])
def test_persistent_spawn_workers_see_each_new_epoch(tmp_path, kind):
    dataset = __datasets(__dataset_root(tmp_path))[kind]
    workers = __loader(dataset, 2)
    first = __epoch_draws(dataset, workers, 0)
    second = __epoch_draws(dataset, workers, 1)

    resumed = __epoch_draws(dataset, __loader(dataset, 2), 1)
    single = __epoch_draws(dataset, __loader(dataset, 0), 1)

    assert first != second
    assert resumed == second
    assert single == second
