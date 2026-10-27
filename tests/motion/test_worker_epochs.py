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
        timeout=60 if workers else 0,
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


@pytest.mark.parametrize("split", ["train", "val"])
def test_each_window_is_a_normalized_slice_of_its_clip(tmp_path, split):
    root = __dataset_root(tmp_path)
    (root / "val.txt").write_text(
        (root / "train.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    np.save(root / "Mean.npy", np.full(263, 0.5, np.float32))
    np.save(root / "Std.npy", np.full(263, 2.0, np.float32))
    data = MotionDataConfig(mirror_augment=False)
    dataset = MotionWindowDataset(root, data, split, 32, seed=1)
    dataset.set_epoch(3)
    for index in range(CLIPS):
        raw = np.load(root / "new_joint_vecs" / f"{index:06d}.npy")
        windows = [(raw[start : start + 32] - 0.5) / 2.0 for start in range(raw.shape[0] - 31)]
        features = dataset[index].features.numpy()
        matches = [
            start for start, window in enumerate(windows) if np.array_equal(window, features)
        ]
        assert matches, index
        if split == "val":
            assert matches == [(raw.shape[0] - 32) // 2]
