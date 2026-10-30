import json

import numpy as np
import pytest
import torch

from text2motion.motion import preparation
from text2motion.motion.contracts import PreparationRequest

BROKEN_FRAMES = 30


def __fake_encode(joints, params):
    if joints.shape[0] == BROKEN_FRAMES:
        raise ValueError("broken clip")
    return np.ones((joints.shape[0] - 1, 263), np.float32), None, None, None


def __layout(tmp_path, monkeypatch):
    layout = preparation.Hml3dPreparationLayout(tmp_path)
    layout.mkdirs()
    np.save(layout.joints / "000021.npy", np.zeros((40, 22, 3), np.float32))
    np.save(layout.joints / "000022.npy", np.zeros((BROKEN_FRAMES, 22, 3), np.float32))
    monkeypatch.setattr(preparation, "target_offsets", lambda reference: None)
    monkeypatch.setattr(preparation, "feature_params", lambda offsets: None)
    monkeypatch.setattr(preparation, "encode_joints", __fake_encode)
    monkeypatch.setattr(
        preparation,
        "recover_from_ric",
        lambda data, joints: torch.zeros((1, data.shape[1], joints, 3)),
    )
    return layout


def __manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_feature_stage_writes_the_good_clips_then_stops_with_the_failed_names(
    tmp_path, monkeypatch
):
    layout = __layout(tmp_path, monkeypatch)
    request = PreparationRequest(out_dir=tmp_path)

    with pytest.raises(RuntimeError, match="000022.npy"):
        preparation.encode_joint_clips_to_hml3d263(request, layout)

    manifest = __manifest(tmp_path / preparation.FEATURES_MANIFEST)
    assert manifest["failed_names"] == ["000022.npy"]
    assert manifest["complete"] is False
    assert manifest["written"] == 1
    assert (layout.new_joint_vecs / "000021.npy").is_file()
    assert not (layout.new_joint_vecs / "000022.npy").exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_the_index_stage_records_clips_without_a_pose_file(tmp_path, monkeypatch):
    layout = __layout(tmp_path, monkeypatch)
    pose = tmp_path / "walk.npy"
    np.save(pose, np.zeros((50, 22, 3), np.float32))
    index = tmp_path / "index.csv"
    index.write_text(
        "source_path,new_name,start_frame,end_frame\n"
        "./pose_data/ACCAD/walk.npy,000100.npy,0,20\n"
        "./pose_data/ACCAD/gone.npy,000101.npy,0,20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preparation,
        "resolve_pose_path",
        lambda source, root: pose if "walk" in source else None,
    )
    request = PreparationRequest(out_dir=tmp_path, index_csv=index)

    preparation.build_indexed_joint_clips(request, layout)

    manifest = __manifest(tmp_path / preparation.INDEX_MANIFEST)
    assert manifest["missing_names"] == ["000101.npy"]
    assert manifest["written"] == 1
    assert (layout.joints / "000100.npy").is_file()
    assert (layout.joints / "M000100.npy").is_file()


def test_the_index_stage_refuses_bad_crop_bounds_and_marks_missing_files(tmp_path, monkeypatch):
    layout = __layout(tmp_path, monkeypatch)
    pose = tmp_path / "walk.npy"
    np.save(pose, np.zeros((50, 22, 3), np.float32))
    index = tmp_path / "index.csv"
    index.write_text(
        "source_path,new_name,start_frame,end_frame\n"
        "./pose_data/ACCAD/walk.npy,000100.npy,0,20\n"
        "./pose_data/ACCAD/walk.npy,000102.npy,-1,20\n"
        "./pose_data/ACCAD/walk.npy,000103.npy,20,20\n"
        "./pose_data/ACCAD/walk.npy,000104.npy,49,60\n"
        "./pose_data/ACCAD/gone.npy,000101.npy,0,20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preparation, "resolve_pose_path", lambda source, root: pose if "walk" in source else None
    )
    request = PreparationRequest(out_dir=tmp_path, index_csv=index)

    with pytest.raises(ValueError, match="bad crop bounds"):
        preparation.build_indexed_joint_clips(request, layout)

    manifest = __manifest(tmp_path / preparation.INDEX_MANIFEST)
    assert manifest["invalid_names"] == ["000102.npy", "000103.npy", "000104.npy"]
    assert manifest["missing_names"] == ["000101.npy"]
    assert manifest["complete"] is False
    assert (layout.joints / "000100.npy").is_file()
