import numpy as np
import pytest
import torch

from text2motion.motion import representation
from text2motion.motion.contracts import MotionDataConfig, Split
from text2motion.motion.datasets import (
    CaptionedMotionDataset,
    MotionWindowDataset,
    collate_motion_clips,
)
from text2motion.motion.model import MotionClip
from text2motion.motion.normalization import MotionScaler, load_clips
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


@pytest.mark.parametrize(
    ("mean", "std", "match"),
    [
        (np.zeros(3, np.float32), np.ones(4, np.float32), "shape mismatch"),
        (np.zeros(3, np.float32), np.array([1.0, 0.0, 1.0], np.float32), "zero or negative"),
        (np.array([0.0, np.nan, 0.0], np.float32), np.ones(3, np.float32), "not finite"),
    ],
)
def test_a_scaler_built_directly_rejects_bad_statistics(mean, std, match):
    with pytest.raises(ValueError, match=match):
        MotionScaler(mean=mean, std=std)


@pytest.mark.parametrize(
    ("features", "error"),
    [
        (np.ones((2, 3), np.int64), TypeError),
        (torch.ones(2, 3, dtype=torch.int64), TypeError),
        (np.ones((2, 1), np.float32), ValueError),
    ],
    ids=["numpy_integers", "torch_integers", "broadcast_width"],
)
def test_the_scaler_rejects_integer_or_other_width_features(features, error):
    scaler = MotionScaler(mean=np.zeros(3, np.float32), std=np.full(3, 2.0, np.float32))
    with pytest.raises(error):
        scaler.normalize(features)


def test_recovery_rejects_another_width_and_non_finite_values():
    with pytest.raises(ValueError, match="263"):
        representation.recover_from_ric(torch.zeros(4, 272), 22)
    features = np.zeros((4, DIM), np.float32)
    features[1, 7] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        representation.recover_skeleton(features)
    with pytest.raises(ValueError, match="NaN or Inf"):
        representation.StreamingSkeletonRecovery()(features)


def test_encoding_one_frame_raises(synthetic_joints):
    params = representation.feature_params(representation.target_offsets(synthetic_joints))
    with pytest.raises(ValueError, match="frames >= 2"):
        representation.encode_joints(synthetic_joints[:1], params)


def test_float64_recovery_keeps_its_dtype_and_matches_float32():
    features = torch.from_numpy(np.random.default_rng(2).standard_normal((6, DIM)))

    double = representation.recover_from_ric(features, 22)
    single = representation.recover_from_ric(features.float(), 22)

    assert double.dtype == torch.float64
    torch.testing.assert_close(double.float(), single, atol=1e-5, rtol=1e-5)


def test_streamed_recovery_accepts_cuda_features():
    assert torch.cuda.is_available(), "this test checks the CUDA path; this run has no CUDA"
    features = torch.from_numpy(np.random.default_rng(3).standard_normal((8, DIM))).float()

    on_cpu = representation.StreamingSkeletonRecovery()(features)
    on_cuda = representation.StreamingSkeletonRecovery()(features.cuda())

    np.testing.assert_allclose(on_cuda, on_cpu, atol=1e-5)


@pytest.mark.parametrize(
    ("bad", "error", "match"),
    [
        (None, FileNotFoundError, "no feature file"),
        (np.array([[np.nan, 0.0]], np.float32), ValueError, "NaN or Inf"),
        (np.array([[np.inf, 0.0]], np.float32), ValueError, "NaN or Inf"),
        (np.zeros((0, 2), np.float32), ValueError, "empty"),
    ],
    ids=["missing", "nan", "inf", "empty"],
)
def test_fitting_statistics_refuses_a_missing_or_bad_train_clip(tmp_path, bad, error, match):
    np.save(tmp_path / "good.npy", np.ones((3, 2), np.float32))
    if bad is not None:
        np.save(tmp_path / "bad.npy", bad)

    with pytest.raises(error, match=match):
        load_clips(tmp_path, ["good", "bad"])
