import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from text2motion.motion import preparation
from text2motion.motion.normalization import MotionScaler
from text2motion.tokenization.contracts import TokenizeRequest
from text2motion.tokenization.corpus import (
    PACK_IDENTITY_KEY,
    build_token_pack,
    pack_clip_keys,
    read_pack_identity,
    window_starts,
)
from text2motion.tokenization.nn_model import FrozenMotionTokenizer


@pytest.mark.parametrize(
    ("frames", "starts"),
    [(40, [0, 16, 24]), (32, [0, 16]), (16, [0]), (10, [])],
    ids=["tail_end_aligned", "exact_fit", "one_window", "too_short"],
)
def test_windows_cover_every_frame_with_an_end_aligned_last_window(frames, starts):
    assert window_starts(frames, 16, 16) == starts


def __fake_tokenizer() -> FrozenMotionTokenizer:
    def encode(clip):
        steps = clip.frame_count // 4
        return SimpleNamespace(indices=torch.zeros(steps, 2, dtype=torch.long), token_count=steps)

    return cast(FrozenMotionTokenizer, SimpleNamespace(encode=encode))


def test_the_pack_skips_excluded_clips_and_records_its_identity(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    np.save(features / "kept.npy", np.zeros((40, 263), np.float32))
    np.save(features / "heldout.npy", np.zeros((40, 263), np.float32))
    request = TokenizeRequest(
        config=tmp_path / "unused.yaml",
        features_dir=features,
        token_pack=tmp_path / "pack.npz",
        segment_frames=16,
        stride=16,
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
    request = TokenizeRequest(
        config=tmp_path / "unused.yaml",
        features_dir=features,
        token_pack=tmp_path / "pack.npz",
        segment_frames=16,
        stride=16,
    )
    scaler = cast(MotionScaler, SimpleNamespace(normalize=lambda feat: feat))
    build_token_pack(__fake_tokenizer(), scaler, request, set(), {"tokenizer_sha256": "abc"})
    spec = importlib.util.spec_from_file_location(
        "scaling_budget", Path("scripts/benchmarks/scaling_budget.py")
    )
    assert spec is not None
    assert spec.loader is not None
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


@pytest.mark.parametrize("case", ["all_excluded", "all_too_short"])
def test_a_pack_without_any_window_is_not_written(tmp_path, case):
    features = tmp_path / "features"
    features.mkdir()
    frames = 8 if case == "all_too_short" else 40
    np.save(features / "only.npy", np.zeros((frames, 263), np.float32))
    request = TokenizeRequest(
        config=tmp_path / "unused.yaml",
        features_dir=features,
        token_pack=tmp_path / "pack.npz",
        segment_frames=16,
        stride=16,
    )
    scaler = cast(MotionScaler, SimpleNamespace(normalize=lambda feat: feat))
    excluded = {"only.npy"} if case == "all_excluded" else set()

    with pytest.raises(ValueError, match="would be empty"):
        build_token_pack(__fake_tokenizer(), scaler, request, excluded, {})
    assert not (tmp_path / "pack.npz").exists()


@pytest.mark.parametrize("field", ["segment_frames", "stride"])
def test_a_tokenize_request_rejects_a_window_or_stride_below_one(tmp_path, field):
    with pytest.raises(ValidationError, match=field):
        TokenizeRequest.model_validate({"config": tmp_path / "unused.yaml", field: 0})
