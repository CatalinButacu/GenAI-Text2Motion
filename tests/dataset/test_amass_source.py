from pathlib import Path
from typing import ClassVar

import numpy as np

from text2motion.dataset.sources.amass import source as amass_source
from text2motion.dataset.sources.amass.source import AmassSource


class FakeExtractor:
    calls: ClassVar[list[Path]] = []

    def __init__(self, smplx_dir: Path, joint_count: int, fps: int, **kwargs) -> None:
        self.joint_count = joint_count
        self.fps = fps

    def extract(self, src_path) -> np.ndarray:
        FakeExtractor.calls.append(Path(src_path))
        return np.zeros((4, self.joint_count, 3), dtype=np.float32)


def test_joints_of_uses_the_amass_extractor_with_22_joints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(amass_source, "AmassPoseExtractor", FakeExtractor)
    FakeExtractor.calls = []
    source = AmassSource(smplx_dir=tmp_path, fps=60)
    joints = source.joints_of(tmp_path / "KIT" / "4" / "a_stageii.npz")
    assert joints is not None
    assert joints.shape == (4, 22, 3)
    assert FakeExtractor.calls == [tmp_path / "KIT" / "4" / "a_stageii.npz"]


def test_extractor_is_created_once_for_many_recordings(tmp_path: Path, monkeypatch) -> None:
    created: list[int] = []

    class Counting(FakeExtractor):
        def __init__(self, *args, **kwargs) -> None:
            created.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(amass_source, "AmassPoseExtractor", Counting)
    source = AmassSource(smplx_dir=tmp_path, fps=60)
    source.joints_of(tmp_path / "a.npz")
    source.joints_of(tmp_path / "b.npz")
    assert len(created) == 1


class RecipeExtractor(FakeExtractor):
    def recipe(self) -> dict:
        return {"joints": self.joint_count, "fps": self.fps}


def test_recipe_comes_from_the_same_extractor(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(amass_source, "AmassPoseExtractor", RecipeExtractor)
    source = AmassSource(smplx_dir=tmp_path, fps=60)
    assert source.recipe() == {"joints": 22, "fps": 60}
