from pathlib import Path

from text2motion.dataset.sources.base import RecordingSource, recording_key
from text2motion.dataset.sources.humanml3d.source import HumanML3DSource
from text2motion.dataset.sources.registry import SOURCES, source_named

KIT = "KIT/4/WalkInCounterClockwiseCircle02"


def test_three_names_of_one_recording_give_one_key() -> None:
    assert recording_key("./pose_data/KIT/4/WalkInCounterClockwiseCircle02_poses.npy") == KIT
    assert recording_key("KIT/4/WalkInCounterClockwiseCircle02_stageii.npz") == KIT
    assert recording_key("KIT/4/WalkInCounterClockwiseCircle02_poses.npz") == KIT


def test_registry_holds_every_source_with_its_own_name() -> None:
    assert set(SOURCES) == {"amass", "humanml3d"}
    for name, cls in SOURCES.items():
        assert issubclass(cls, RecordingSource)
        assert cls.name == name


def test_unknown_source_name_fails_loud() -> None:
    try:
        source_named("babel-typo")
    except KeyError as error:
        assert "babel-typo" in str(error)
    else:
        raise AssertionError("expected KeyError")


def test_humanml3d_split_comes_from_the_official_files(tmp_path: Path) -> None:
    (tmp_path / "index.csv").write_text(
        "source_path,start_frame,end_frame,new_name\n./pose_data/A/a.npy,0,10,000001.npy\n",
        encoding="utf-8",
    )
    (tmp_path / "train.txt").write_text("000001\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("000002\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("", encoding="utf-8")
    source = HumanML3DSource(tmp_path)
    assert source.split_of("000001") == "train"
    assert source.split_of("000002") == "val"
    assert source.split_of("999999") is None
