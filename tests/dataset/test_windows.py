import numpy as np
import pytest

from text2motion.dataset.core.windows import CropError, cut_window


def frames(count: int) -> np.ndarray:
    data = np.zeros((count, 22, 3), dtype=np.float32)
    data[..., 0] = np.arange(count)[:, None]
    return data


def test_cut_takes_the_bounds_after_the_head_trim() -> None:
    clip = cut_window(frames(20), start=2, end=6, trim=3)
    assert clip.shape == (4, 22, 3)
    assert clip[0, 0, 0] == -5.0


def test_cut_flips_the_x_axis_once() -> None:
    clip = cut_window(frames(10), start=0, end=4, trim=0)
    assert clip[1, 0, 0] == -1.0


def test_bad_bounds_are_refused() -> None:
    with pytest.raises(CropError, match="bad crop bounds"):
        cut_window(frames(10), start=5, end=5, trim=0)


def test_too_few_frames_are_refused() -> None:
    with pytest.raises(CropError, match="fewer than 2 frames"):
        cut_window(frames(10), start=0, end=1, trim=0)


def test_humanact12_is_kept_whole_and_not_flipped() -> None:
    clip = cut_window(frames(10), start=0, end=0, trim=0, whole=True)
    assert clip.shape == (10, 22, 3)
    assert clip[1, 0, 0] == 1.0
