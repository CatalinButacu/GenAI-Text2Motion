from pathlib import Path

import numpy as np

from text2motion.dataset.core.scan import check_clip, check_clip_folder


def test_a_good_clip_is_kept(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.ones((50, 263), dtype=np.float32))
    assert check_clip(tmp_path / "a.npy", min_frames=40)["action"] == "keep"


def test_a_non_finite_clip_is_excluded(tmp_path: Path) -> None:
    bad = np.ones((50, 263), dtype=np.float32)
    bad[3, 5] = np.nan
    np.save(tmp_path / "a.npy", bad)
    assert check_clip(tmp_path / "a.npy", min_frames=40)["action"] == "exclude: non-finite values"


def test_a_short_clip_is_filtered_not_excluded(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.ones((10, 263), dtype=np.float32))
    assert check_clip(tmp_path / "a.npy", min_frames=40)["action"] == "filter by min length"


def test_a_wrong_shape_is_excluded(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.ones((50, 10), dtype=np.float32))
    assert check_clip(tmp_path / "a.npy", min_frames=40)["action"] == "exclude: wrong shape"


def test_scan_folder_writes_one_row_per_clip(tmp_path: Path) -> None:
    folder = tmp_path / "vec"
    folder.mkdir()
    np.save(folder / "a.npy", np.ones((50, 263), dtype=np.float32))
    np.save(folder / "b.npy", np.ones((10, 263), dtype=np.float32))
    log = tmp_path / "log.csv"
    counts = check_clip_folder(folder, log, min_frames=40)
    assert counts == {"keep": 1, "filter by min length": 1}
    assert len(log.read_text(encoding="utf-8").splitlines()) == 3
