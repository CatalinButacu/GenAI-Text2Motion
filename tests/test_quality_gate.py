"""Tests for scripts/data/quality_gate.py"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.data.quality_gate import (
    check_coord_system,
    check_files_exist,
    check_no_duplicates,
    check_sample_quality,
    check_stats,
    run_gate,
)

POSE_DIM = 168


def makeClip(tmpDir, name: str, frames: int = 32, poseDim: int = POSE_DIM, hasNan: bool = False):
    arr = np.random.default_rng(0).random((frames, poseDim)).astype(np.float32)
    arr[:, 5] = 0.9  # pelvis Z positive (Y-up)
    if hasNan:
        arr[0, 0] = float("nan")
    np.save(str(tmpDir / f"{name}.npy"), arr)
    return arr


def makeStats(statsDir):
    statsDir.mkdir(parents=True, exist_ok=True)
    np.save(str(statsDir / "mean.npy"), np.zeros(POSE_DIM, dtype=np.float32))
    np.save(str(statsDir / "std.npy"), np.ones(POSE_DIM, dtype=np.float32))


class TestCheckFilesExist:
    def test_passes_when_files_present(self, tmp_path):
        makeClip(tmp_path, "clip_0")
        result = check_files_exist(str(tmp_path))
        assert result["passed"]

    def test_fails_empty_dir(self, tmp_path):
        result = check_files_exist(str(tmp_path))
        assert not result["passed"]


class TestCheckSampleQuality:
    def test_passes_clean_data(self, tmp_path):
        for i in range(5):
            makeClip(tmp_path, f"clip_{i}")
        result = check_sample_quality(str(tmp_path), sampleSize=10, seed=42)
        assert result["passed"]

    def test_fails_nan_data(self, tmp_path):
        for i in range(5):
            makeClip(tmp_path, f"clip_{i}", hasNan=True)
        result = check_sample_quality(str(tmp_path), sampleSize=10, seed=42)
        assert not result["passed"]

    def test_fails_wrong_pose_dim(self, tmp_path):
        arr = np.zeros((32, 99), dtype=np.float32)
        np.save(str(tmp_path / "bad.npy"), arr)
        result = check_sample_quality(str(tmp_path), sampleSize=10, seed=42)
        assert not result["passed"]

    def test_fails_too_short(self, tmp_path):
        makeClip(tmp_path, "short", frames=4)
        result = check_sample_quality(str(tmp_path), sampleSize=10, seed=42)
        assert not result["passed"]


class TestCheckStats:
    def test_passes_valid_stats(self, tmp_path):
        makeStats(tmp_path)
        result = check_stats(str(tmp_path))
        assert result["passed"]

    def test_fails_missing_mean(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        np.save(str(tmp_path / "std.npy"), np.ones(POSE_DIM))
        result = check_stats(str(tmp_path))
        assert not result["passed"]

    def test_fails_wrong_shape(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        np.save(str(tmp_path / "mean.npy"), np.zeros(100))
        np.save(str(tmp_path / "std.npy"), np.ones(100))
        result = check_stats(str(tmp_path))
        assert not result["passed"]


class TestCheckNoDuplicates:
    def test_passes_unique_stems(self, tmp_path):
        for i in range(3):
            makeClip(tmp_path, f"clip_{i}")
        result = check_no_duplicates(str(tmp_path))
        assert result["passed"]

    def test_detects_duplicates(self, tmp_path):
        sub1 = tmp_path / "a"
        sub2 = tmp_path / "b"
        sub1.mkdir()
        sub2.mkdir()
        makeClip(sub1, "clip_0")
        makeClip(sub2, "clip_0")  # same stem in different subdirs
        result = check_no_duplicates(str(tmp_path))
        assert not result["passed"]


class TestRunGate:
    def test_passes_clean_setup(self, tmp_path):
        dataDir = tmp_path / "data"
        dataDir.mkdir()
        statsDir = tmp_path / "stats"
        for i in range(5):
            makeClip(dataDir, f"clip_{i}")
        makeStats(statsDir)
        result = run_gate(str(dataDir), str(statsDir), sampleSize=5, seed=42)
        assert result["passed"]
        assert all(c["passed"] for c in result["checks"])

    def test_fails_nan_data(self, tmp_path):
        dataDir = tmp_path / "data"
        dataDir.mkdir()
        statsDir = tmp_path / "stats"
        for i in range(3):
            makeClip(dataDir, f"clip_{i}", hasNan=True)
        makeStats(statsDir)
        result = run_gate(str(dataDir), str(statsDir), sampleSize=5, seed=42)
        assert not result["passed"]

    def test_saves_json_report(self, tmp_path):
        dataDir = tmp_path / "data"
        dataDir.mkdir()
        statsDir = tmp_path / "stats"
        for i in range(3):
            makeClip(dataDir, f"clip_{i}")
        makeStats(statsDir)
        outFile = str(tmp_path / "report.json")
        run_gate(str(dataDir), str(statsDir), sampleSize=5, seed=42, outputPath=outFile)
        loaded = json.loads((tmp_path / "report.json").read_text())
        assert "passed" in loaded
        assert "checks" in loaded
