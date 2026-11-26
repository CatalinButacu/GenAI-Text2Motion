import pytest

from text2motion.dataset.contracts import MotionDataConfig


def test_default_source_is_humanml3d() -> None:
    assert MotionDataConfig().sources == ("humanml3d",)


def test_registered_source_names_are_accepted() -> None:
    assert MotionDataConfig(sources=("amass", "humanml3d")).sources == ("amass", "humanml3d")


def test_unknown_source_name_fails_loud() -> None:
    with pytest.raises(ValueError, match="babel-typo"):
        MotionDataConfig(sources=("babel-typo",))
