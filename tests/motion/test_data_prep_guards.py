import numpy as np
import pytest

from text2motion.dataset.datasets import item_rng
from text2motion.dataset.preparation import run_hml3d_preparation
from text2motion.motion.contracts import PreparationRequest, PreparationStage
from text2motion.motion.normalization import fit_train_stats, normalization_stats
from text2motion.motion.storage import listed_clip_ids

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


def test_a_missing_split_file_stops_the_listed_reader(tmp_path):
    with pytest.raises(FileNotFoundError, match="every listed id is needed"):
        listed_clip_ids(tmp_path / "train.txt")


def test_stats_refuse_to_fit_without_a_train_split(tmp_path):
    with pytest.raises(FileNotFoundError, match="fitted on the train split alone"):
        run_hml3d_preparation(PreparationRequest(out_dir=tmp_path), PreparationStage.STATS)


def test_stats_ignore_val_and_test_clips(tmp_path):
    vec_dir = build_dirs(tmp_path)
    write_clip(vec_dir, "000001", 1.0)
    write_clip(vec_dir, "000002", 1.0)
    write_clip(vec_dir, "000999", 1000.0)
    (tmp_path / "train.txt").write_text("000001\n000002\n", encoding="utf-8")

    names = listed_clip_ids(tmp_path / "train.txt")
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
    a = item_rng(2026, 0, 5).random()
    b = item_rng(2026, 0, 5).random()
    assert a == b

    assert item_rng(2026, 1, 5).random() != a
    assert item_rng(2026, 0, 6).random() != a
    assert item_rng(7, 0, 5).random() != a


def test_item_rng_does_not_depend_on_call_order():
    forward = [item_rng(2026, 3, i).randint(0, 10_000) for i in range(20)]
    backward = [item_rng(2026, 3, i).randint(0, 10_000) for i in reversed(range(20))][::-1]
    assert forward == backward


def test_foot_contact_limit_keeps_the_same_speed_at_60_fps():
    import torch

    from text2motion.motion.representation import feature_params

    offsets = torch.zeros(JOINTS, 3)
    at_20 = feature_params(offsets).foot_velocity_threshold
    at_60 = feature_params(offsets, 60).foot_velocity_threshold
    assert at_60 == pytest.approx(at_20 / 9)
    with pytest.raises(ValueError, match="positive"):
        feature_params(offsets, 0)


def test_index_frames_scale_to_a_multiple_of_20_fps_only():
    from text2motion.dataset.preparation import index_frame_scale

    assert index_frame_scale(20) == 1
    assert index_frame_scale(60) == 3
    with pytest.raises(ValueError, match="fps=30"):
        index_frame_scale(30)


@pytest.mark.parametrize(
    ("start", "end"),
    [((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))],
)
def test_qbetween_turns_a_vector_onto_its_exact_opposite(start, end):
    import torch

    from text2motion.motion.geometry import qbetween, qrot

    v0 = torch.tensor([start])
    v1 = torch.tensor([end])
    q = qbetween(v0, v1)
    assert torch.isfinite(q).all()
    assert torch.allclose(qrot(q, v0), v1, atol=1e-6)


def test_qbetween_is_unchanged_for_vectors_that_are_not_opposite():
    import torch

    from text2motion.motion.geometry import qbetween, qrot

    v0 = torch.tensor([[0.0068, 0.0, -0.99998], [0.6, 0.0, 0.8]])
    v1 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    rotated = qrot(qbetween(v0, v1), v0 / v0.norm(dim=-1, keepdim=True))
    assert torch.allclose(rotated, v1, atol=1e-4)


def test_normalization_stats_streamed_equals_one_concatenated_array():
    rng = np.random.default_rng(2026)
    clips = [rng.normal(3.0, 2.0, (frames, DIM)).astype(np.float32) for frames in (7, 40, 1, 123)]
    mean, std = normalization_stats(iter(clips), JOINTS)
    data = np.concatenate(clips, axis=0).astype(np.float64)
    ric_end = 4 + (JOINTS - 1) * 3
    assert np.allclose(mean, data.mean(axis=0), atol=1e-5)
    assert np.isclose(std[0], data.std(axis=0)[0], atol=1e-5)
    assert np.isclose(std[4], data.std(axis=0)[4:ric_end].mean(), atol=1e-5)
