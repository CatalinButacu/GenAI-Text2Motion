import json

import numpy as np
import pytest

from text2motion.dataset import preparation
from text2motion.motion.contracts import PreparationRequest

CLIPS = ("walk", "run", "jump")


class FakeExtractor:
    fps = 20
    offset = 0.0
    calls: list[str] = []

    def __init__(self, smplx_dir, joint_count, device, fps):
        pass

    def recipe(self):
        return {"fps": FakeExtractor.fps, "joint_count": 22}

    def extract(self, src):
        FakeExtractor.calls.append(src.stem)
        return np.full((4, 22, 3), len(src.stem) + FakeExtractor.offset, np.float32)


def build_layout(tmp_path, monkeypatch):
    amass = tmp_path / "amass" / "ACCAD"
    amass.mkdir(parents=True)
    for clip in CLIPS:
        np.savez(amass / f"{clip}.npz", trans=np.zeros(1))
    FakeExtractor.fps = 20
    FakeExtractor.offset = 0.0
    FakeExtractor.calls = []
    monkeypatch.setattr(preparation, "AmassPoseExtractor", FakeExtractor)
    request = PreparationRequest(
        out_dir=tmp_path / "regen", amass_dir=tmp_path / "amass", smplx_models=tmp_path
    )
    layout = preparation.Hml3dPreparationLayout(tmp_path / "regen")
    layout.mkdirs()
    return request, layout


def read_manifest(layout):
    path = layout.pose_data / preparation.POSE_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))


def test_an_extraction_writes_a_complete_manifest(tmp_path, monkeypatch):
    request, layout = build_layout(tmp_path, monkeypatch)

    counts = preparation.extract_amass_to_y_up_joints(request, layout)

    assert counts == {"written": 3, "reused": 0, "no_motion": 0}
    assert read_manifest(layout)["complete"] is True
    assert read_manifest(layout)["recipe"]["fps"] == 20
    assert not list(layout.pose_data.rglob("*.tmp"))


def test_a_rerun_with_another_recipe_is_refused(tmp_path, monkeypatch):
    request, layout = build_layout(tmp_path, monkeypatch)
    preparation.extract_amass_to_y_up_joints(request, layout)
    FakeExtractor.fps = 30

    with pytest.raises(ValueError, match="fps"):
        preparation.extract_amass_to_y_up_joints(request, layout)


def legacy_folder(request, layout):
    preparation.extract_amass_to_y_up_joints(request, layout)
    (layout.pose_data / preparation.POSE_MANIFEST).unlink()
    FakeExtractor.calls = []


def test_a_legacy_folder_is_refused_without_adoption(tmp_path, monkeypatch):
    request, layout = build_layout(tmp_path, monkeypatch)
    legacy_folder(request, layout)

    with pytest.raises(ValueError, match="legacy folder"):
        preparation.extract_amass_to_y_up_joints(request, layout)


def test_a_matching_legacy_folder_is_adopted_and_kept_on_reruns(tmp_path, monkeypatch):
    request, layout = build_layout(tmp_path, monkeypatch)
    legacy_folder(request, layout)

    counts = preparation.extract_amass_to_y_up_joints(request, layout, adopt=True)
    preparation.extract_amass_to_y_up_joints(request, layout)

    assert counts["reused"] == 3
    assert sorted(FakeExtractor.calls) == sorted(CLIPS)
    assert read_manifest(layout)["complete"] is True
    assert len(read_manifest(layout)["adopted_sample"]) == 3


def test_a_legacy_folder_that_differs_is_not_adopted(tmp_path, monkeypatch):
    request, layout = build_layout(tmp_path, monkeypatch)
    legacy_folder(request, layout)
    FakeExtractor.offset = 1e-3

    with pytest.raises(ValueError, match="differ from a new extraction"):
        preparation.extract_amass_to_y_up_joints(request, layout, adopt=True)

    assert not (layout.pose_data / preparation.POSE_MANIFEST).exists()
