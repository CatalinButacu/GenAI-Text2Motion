import numpy as np
import pytest

from text2motion.data.hml3d.feature import normalization_stats
from text2motion.data.hml3d.stats import fit_train_stats, read_split_names

JOINTS = 22
DIM = 8 + (JOINTS - 1) * 9 + JOINTS * 3


def write_clip(vec_dir, name: str, value: float, frames: int = 12) -> None:
    arr = np.full((frames, DIM), value, dtype=np.float32)
    arr += np.random.default_rng(abs(hash(name)) % 2**32).normal(0, 0.01, arr.shape)
    np.save(vec_dir / f"{name}.npy", arr)


def build_dirs(tmp_path):
    vec_dir = tmp_path / "new_joint_vecs"
    vec_dir.mkdir(parents=True, exist_ok=True)
    return vec_dir


def test_stats_refuse_to_fit_without_a_train_split(tmp_path):
    with pytest.raises(FileNotFoundError, match="train split"):
        read_split_names(tmp_path / "train.txt")


def test_stats_ignore_val_and_test_clips(tmp_path):
    vec_dir = build_dirs(tmp_path)
    write_clip(vec_dir, "000001", 1.0)
    write_clip(vec_dir, "000002", 1.0)
    write_clip(vec_dir, "000999", 1000.0)
    (tmp_path / "train.txt").write_text("000001\n000002\n", encoding="utf-8")

    names = read_split_names(tmp_path / "train.txt")
    mean, _ = fit_train_stats(vec_dir, names, tmp_path, JOINTS)

    assert mean.shape == (DIM,)
    assert abs(float(mean.mean()) - 1.0) < 0.1


def test_normalization_stats_flattens_groups_and_checks_dim():
    clips = [np.random.default_rng(0).normal(0, 1, (30, DIM)).astype(np.float32)]
    _, std = normalization_stats(clips, JOINTS)

    ric_end = 4 + (JOINTS - 1) * 3
    assert np.allclose(std[4:ric_end], std[4])
    assert np.allclose(std[1:3], std[1])

    with pytest.raises(ValueError, match="expected"):
        normalization_stats([np.zeros((5, DIM + 1), np.float32)], JOINTS)


def test_normalization_stats_rejects_empty_input():
    with pytest.raises(ValueError, match="no clips"):
        normalization_stats([], JOINTS)


def test_item_rng_is_deterministic_and_varies_by_epoch_and_index():
    from text2motion.shared.seed import item_rng

    a = item_rng(2026, 0, 5).random()
    b = item_rng(2026, 0, 5).random()
    assert a == b

    assert item_rng(2026, 1, 5).random() != a
    assert item_rng(2026, 0, 6).random() != a
    assert item_rng(7, 0, 5).random() != a


def test_item_rng_does_not_depend_on_call_order():
    from text2motion.shared.seed import item_rng

    forward = [item_rng(2026, 3, i).randint(0, 10_000) for i in range(20)]
    backward = [item_rng(2026, 3, i).randint(0, 10_000) for i in reversed(range(20))][::-1]
    assert forward == backward
