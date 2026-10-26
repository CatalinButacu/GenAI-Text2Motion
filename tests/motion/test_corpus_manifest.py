import json

import numpy as np
import pytest

from text2motion.motion import preparation
from text2motion.motion.contracts import PreparationRequest, Split
from text2motion.motion.storage import FEATURES_DIR, SPLIT_FILES

BROKEN_FRAMES = 30


def __fake_encode(joints, params):
    if joints.shape[0] == BROKEN_FRAMES:
        raise ValueError("broken capture")
    return np.ones((joints.shape[0] - 1, 263), np.float32), None, None, None


def __layout(tmp_path, monkeypatch, heldout=frozenset()):
    regen = tmp_path / "regen"
    (regen / "joints").mkdir(parents=True)
    np.save(regen / "joints" / "000021.npy", np.zeros((5, 22, 3), np.float32))
    pose_folder = regen / "pose_data" / "ACCAD"
    pose_folder.mkdir(parents=True)
    np.save(pose_folder / "walk_stageii.npy", np.zeros((40, 22, 3), np.float32))
    np.save(pose_folder / "run_stageii.npy", np.zeros((50, 22, 3), np.float32))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split in (Split.VALIDATION, Split.TEST):
        (dataset / SPLIT_FILES[split]).write_text("000001\n", encoding="utf-8")
    index_csv = tmp_path / "index.csv"
    index_csv.write_text("source_path,new_name\n", encoding="utf-8")
    monkeypatch.setattr(preparation, "target_offsets", lambda reference: None)
    monkeypatch.setattr(preparation, "feature_params", lambda offsets: None)
    monkeypatch.setattr(preparation, "encode_joints", __fake_encode)
    monkeypatch.setattr(preparation, "heldout_feature_names", lambda *args: set(heldout))
    request = PreparationRequest(out_dir=dataset, index_csv=index_csv)
    return request, regen, tmp_path / "corpus"


def __manifest(out_dir):
    return json.loads((out_dir / preparation.CORPUS_MANIFEST).read_text(encoding="utf-8"))


def test_a_build_writes_a_complete_manifest_with_the_recipe(tmp_path, monkeypatch):
    request, regen, out_dir = __layout(tmp_path, monkeypatch)

    summary = preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)

    manifest = __manifest(out_dir)
    assert summary["written"] == 2
    assert manifest["complete"] is True
    assert manifest["failed_names"] == []
    assert manifest["recipe"]["min_frames"] == 10
    assert manifest["recipe"]["mirror_x"] is True
    assert manifest["recipe"]["head_trim_seconds"]["MPI_HDM05"] == 3.0


def test_a_rerun_counts_the_frames_of_reused_files(tmp_path, monkeypatch):
    request, regen, out_dir = __layout(tmp_path, monkeypatch)
    first = preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)

    second = preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)

    assert second["reused"] == 2
    assert second["written"] == 0
    assert second["frames"] == first["frames"] == 39 + 49


def test_a_rerun_with_another_recipe_is_refused(tmp_path, monkeypatch):
    request, regen, out_dir = __layout(tmp_path, monkeypatch)
    preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)

    with pytest.raises(ValueError, match="min_frames"):
        preparation.build_amass_pretraining_corpus(request, regen, out_dir, 20)


def test_a_folder_with_features_and_no_manifest_is_refused(tmp_path, monkeypatch):
    request, regen, out_dir = __layout(tmp_path, monkeypatch)
    (out_dir / FEATURES_DIR).mkdir(parents=True)
    np.save(out_dir / FEATURES_DIR / "ACCAD__walk_stageii.npy", np.zeros((3, 263), np.float32))

    with pytest.raises(ValueError, match="legacy corpus"):
        preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)


def test_a_heldout_file_already_in_the_folder_stops_the_build(tmp_path, monkeypatch):
    request, regen, out_dir = __layout(tmp_path, monkeypatch)
    preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)
    monkeypatch.setattr(
        preparation, "heldout_feature_names", lambda *args: {"ACCAD__run_stageii.npy"}
    )

    with pytest.raises(ValueError, match="heldout files are already in"):
        preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)


def test_heldout_sources_are_skipped(tmp_path, monkeypatch):
    request, regen, out_dir = __layout(tmp_path, monkeypatch, frozenset({"ACCAD__run_stageii.npy"}))

    summary = preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)

    assert summary["skipped_heldout"] == 1
    assert not (out_dir / FEATURES_DIR / "ACCAD__run_stageii.npy").exists()


def test_failures_finish_the_build_then_raise_with_the_names(tmp_path, monkeypatch):
    request, regen, out_dir = __layout(tmp_path, monkeypatch)
    np.save(regen / "pose_data" / "ACCAD" / "bad_stageii.npy", np.zeros((30, 22, 3)))

    with pytest.raises(RuntimeError, match="1 source files failed"):
        preparation.build_amass_pretraining_corpus(request, regen, out_dir, 10)

    manifest = __manifest(out_dir)
    assert manifest["complete"] is False
    assert manifest["failed_names"] == ["ACCAD__bad_stageii.npy"]
    assert manifest["written"] == 2
