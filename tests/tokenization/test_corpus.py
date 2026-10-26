import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from text2motion.motion import preparation
from text2motion.motion.normalization import MotionScaler
from text2motion.tokenization.contracts import TokenPackRequest
from text2motion.tokenization.corpus import (
    PACK_IDENTITY_KEY,
    build_token_pack,
    pack_clip_keys,
    read_pack_identity,
    window_starts,
)
from text2motion.tokenization.nn_model import MotionTokenizer


@pytest.mark.parametrize(
    ("frames", "starts"),
    [(40, [0, 16, 24]), (32, [0, 16]), (16, [0]), (10, [])],
    ids=["tail_end_aligned", "exact_fit", "one_window", "too_short"],
)
def test_windows_cover_every_frame_with_an_end_aligned_last_window(frames, starts):
    assert window_starts(frames, 16, 16) == starts


def __fake_tokenizer() -> MotionTokenizer:
    def encode(clip):
        steps = clip.frame_count // 4
        return SimpleNamespace(indices=torch.zeros(steps, 2, dtype=torch.long), token_count=steps)

    return cast(MotionTokenizer, SimpleNamespace(encode=encode))


def test_the_pack_skips_excluded_clips_and_records_its_identity(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    np.save(features / "kept.npy", np.zeros((40, 263), np.float32))
    np.save(features / "heldout.npy", np.zeros((40, 263), np.float32))
    request = TokenPackRequest(
        features_dir=features, out_path=tmp_path / "pack.npz", segment_frames=16, stride=16
    )
    scaler = cast(MotionScaler, SimpleNamespace(normalize=lambda feat: feat))

    summary = build_token_pack(
        __fake_tokenizer(), scaler, request, {"heldout.npy"}, {"tokenizer_sha256": "abc"}
    )

    with np.load(tmp_path / "pack.npz") as pack:
        windows = sorted(key for key in pack.files if key != PACK_IDENTITY_KEY)
    identity = read_pack_identity(tmp_path / "pack.npz")
    assert windows == ["kept__0", "kept__16", "kept__24"]
    assert summary["excluded"] == 1
    assert identity["tokenizer_sha256"] == "abc"
    assert identity["excluded_present"] == 1
    assert identity["tail"] == "end_aligned"


def test_pack_readers_count_only_clip_windows(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    np.save(features / "kept.npy", np.zeros((40, 263), np.float32))
    request = TokenPackRequest(
        features_dir=features, out_path=tmp_path / "pack.npz", segment_frames=16, stride=16
    )
    scaler = cast(MotionScaler, SimpleNamespace(normalize=lambda feat: feat))
    build_token_pack(__fake_tokenizer(), scaler, request, set(), {"tokenizer_sha256": "abc"})
    spec = importlib.util.spec_from_file_location(
        "scaling_budget", Path("scripts/benchmarks/scaling_budget.py")
    )
    assert spec is not None and spec.loader is not None
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)

    with np.load(tmp_path / "pack.npz") as pack:
        keys = pack_clip_keys(pack)
    windows, tokens = budget.amass_token_file_total(tmp_path / "pack.npz", 2)

    assert sorted(keys) == ["kept__0", "kept__16", "kept__24"]
    assert windows == 3
    assert tokens == 3 * 4 * 2


def test_a_pack_without_an_identity_record_is_refused(tmp_path):
    np.savez(tmp_path / "legacy.npz", clip__0=np.zeros((4, 2), np.int16))

    with pytest.raises(ValueError, match="without an identity record"):
        read_pack_identity(tmp_path / "legacy.npz")


def test_a_copy_of_a_heldout_capture_under_another_name_is_excluded(tmp_path, monkeypatch):
    pose_root = tmp_path / "pose_data"
    (pose_root / "ACCAD").mkdir(parents=True)
    heldout = pose_root / "ACCAD" / "run_stageii.npy"
    np.save(heldout, np.ones((8, 3), np.float32))
    np.save(pose_root / "ACCAD" / "run_a_stageii.npy", np.ones((8, 3), np.float32))
    np.save(pose_root / "ACCAD" / "walk_stageii.npy", np.zeros((8, 3), np.float32))
    monkeypatch.setattr(preparation, "heldout_sources", lambda *args: {heldout.resolve()})

    names = preparation.heldout_feature_names(Path("index.csv"), tmp_path, pose_root)

    assert names == {"ACCAD__run_stageii.npy", "ACCAD__run_a_stageii.npy"}
