import numpy as np
import pytest

from text2motion.motion import normalization
from text2motion.motion.normalization import MotionScaler, fit_train_stats
from text2motion.motion.storage import MEAN_FILE, STD_FILE


def __write_stats(folder, mean, std):
    np.save(folder / MEAN_FILE, np.asarray(mean, np.float32))
    np.save(folder / STD_FILE, np.asarray(std, np.float32))


@pytest.mark.parametrize(
    ("std", "message"),
    [
        ([1.0, 0.0, 1.0], "zero or negative"),
        ([1.0, -2.0, 1.0], "zero or negative"),
        ([1.0, np.nan, 1.0], "not finite"),
        ([1.0, np.inf, 1.0], "not finite"),
    ],
    ids=["zero", "negative", "nan", "inf"],
)
def test_invalid_std_values_are_refused(tmp_path, std, message):
    __write_stats(tmp_path, [0.0, 0.0, 0.0], std)

    with pytest.raises(ValueError, match=message):
        MotionScaler.load(tmp_path)


def test_a_two_dimensional_scaler_is_refused(tmp_path):
    __write_stats(tmp_path, np.zeros((2, 3)), np.ones((2, 3)))

    with pytest.raises(ValueError, match="1-D"):
        MotionScaler.load(tmp_path)


def test_a_valid_scaler_round_trips(tmp_path):
    __write_stats(tmp_path, [1.0, 2.0, 3.0], [0.5, 1.0, 2.0])
    scaler = MotionScaler.load(tmp_path, dim=3)
    features = np.array([[2.0, 4.0, 9.0]], np.float32)

    np.testing.assert_allclose(scaler.denormalize(scaler.normalize(features)), features)


def test_fitted_stats_with_a_zero_std_are_not_written(tmp_path, monkeypatch):
    zero_std = (np.zeros(3, np.float32), np.zeros(3, np.float32))
    monkeypatch.setattr(normalization, "load_clips", lambda vec_dir, names: [np.zeros((4, 3))])
    monkeypatch.setattr(normalization, "normalization_stats", lambda clips, joints: zero_std)

    with pytest.raises(ValueError, match="zero or negative"):
        fit_train_stats(tmp_path, ["000001"], tmp_path, 22)

    assert not (tmp_path / MEAN_FILE).exists()
    assert not (tmp_path / STD_FILE).exists()
