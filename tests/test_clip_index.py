import json

import numpy as np

from text2motion.data.hml3d.clip_index import CACHE_NAME, clip_lengths


def make_clip(vec_dir, name, frames, dim=263):
    np.save(vec_dir / f"{name}.npy", np.zeros((frames, dim), np.float32))


def build(tmp_path):
    vec_dir = tmp_path / "new_joint_vecs"
    vec_dir.mkdir(parents=True)
    return vec_dir


def test_lengths_are_correct_and_cached(tmp_path):
    vec_dir = build(tmp_path)
    make_clip(vec_dir, "a", 40)
    make_clip(vec_dir, "b", 77)

    first = clip_lengths(vec_dir, ["a", "b"])
    assert first == {"a": 40, "b": 77}

    cache_path = tmp_path / CACHE_NAME
    assert cache_path.is_file()
    assert set(json.loads(cache_path.read_text())) == {"a", "b"}

    assert clip_lengths(vec_dir, ["a", "b"]) == first


def test_missing_clips_are_skipped_not_invented(tmp_path):
    vec_dir = build(tmp_path)
    make_clip(vec_dir, "a", 40)

    assert clip_lengths(vec_dir, ["a", "ghost"]) == {"a": 40}


def test_cache_is_invalidated_when_a_clip_changes_size(tmp_path):
    vec_dir = build(tmp_path)
    make_clip(vec_dir, "a", 40)
    assert clip_lengths(vec_dir, ["a"]) == {"a": 40}

    make_clip(vec_dir, "a", 120)
    assert clip_lengths(vec_dir, ["a"]) == {"a": 120}


def test_stale_cache_entry_is_recomputed(tmp_path):
    vec_dir = build(tmp_path)
    make_clip(vec_dir, "a", 40)
    clip_lengths(vec_dir, ["a"])

    cache_path = tmp_path / CACHE_NAME
    cache_path.write_text(json.dumps({"a": [9999, 1]}), encoding="utf-8")

    assert clip_lengths(vec_dir, ["a"]) == {"a": 40}
