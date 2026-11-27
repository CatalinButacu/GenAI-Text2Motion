from pathlib import Path

import pytest

from text2motion.dataset.splits.plan import SplitPlan
from text2motion.dataset.splits.train_access import TrainView


def make_plan(folder: Path) -> SplitPlan:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.csv").write_text(
        "source_path,start_frame,end_frame,new_name\n"
        "./pose_data/A/a.npy,0,10,000001.npy\n"
        "./pose_data/B/b.npy,0,10,000002.npy\n"
        "./pose_data/C/c.npy,0,10,000003.npy\n"
        "./pose_data/B/b.npy,0,10,000004.npy\n"
        "./pose_data/D/d.npy,0,10,000005.npy\n",
        encoding="utf-8",
    )
    (folder / "train.txt").write_text("000001\n000004\n000005\n", encoding="utf-8")
    (folder / "val.txt").write_text("000002\n", encoding="utf-8")
    (folder / "test.txt").write_text("000003\n", encoding="utf-8")
    return SplitPlan.from_official(folder)


def test_plan_protects_recordings_of_val_and_test(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    assert plan.protected_sources == frozenset({"./pose_data/B/b.npy", "./pose_data/C/c.npy"})


def test_plan_training_names_exclude_protected_sources(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    assert plan.training_names() == frozenset({"000001", "000005"})


def test_view_allows_training_clips_only(tmp_path: Path) -> None:
    view = TrainView(make_plan(tmp_path))
    assert view.allows("000001")
    assert not view.allows("000002")
    assert not view.allows("000003")
    assert not view.allows("000004")


def test_view_refuses_val_and_test_access(tmp_path: Path) -> None:
    view = TrainView(make_plan(tmp_path))
    with pytest.raises(PermissionError, match="000002"):
        view.require("000002")
    assert view.require("000001") == "000001"


def test_view_cannot_be_widened(tmp_path: Path) -> None:
    view = TrainView(make_plan(tmp_path))
    assert not hasattr(view, "add")
    assert isinstance(view.names(), frozenset)


def test_training_ids_come_from_the_allowed_list(tmp_path: Path) -> None:
    from text2motion.dataset.splits.train_access import split_clip_ids
    from text2motion.motion.contracts import Split

    (tmp_path / "train.txt").write_text("000001\n000004\n000005\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("000002\n", encoding="utf-8")
    (tmp_path / "train_allowed.txt").write_text("000001\n000005\n", encoding="utf-8")
    assert split_clip_ids(tmp_path, Split.TRAIN) == ["000001", "000005"]
    assert split_clip_ids(tmp_path, Split.VALIDATION) == ["000002"]


def test_training_without_the_allowed_list_fails_loud(tmp_path: Path) -> None:
    import pytest as _pytest

    from text2motion.dataset.splits.train_access import split_clip_ids
    from text2motion.motion.contracts import Split

    (tmp_path / "train.txt").write_text("000001\n", encoding="utf-8")
    with _pytest.raises(FileNotFoundError, match="train_allowed"):
        split_clip_ids(tmp_path, Split.TRAIN)
