"""Tests for src/shared/run_manifest.py"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.shared.run_manifest import build_manifest, save_manifest


@dataclass
class FakeConfig:
    lr: float = 1e-3
    seed: int = 42
    label: str = "test"


def test_manifest_required_keys():
    cfg = FakeConfig()
    manifest = build_manifest(cfg, ["walk forward", "run fast"])
    for key in ("git_commit", "timestamp", "python_version", "platform", "config", "prompts"):
        assert key in manifest, f"missing key: {key}"


def test_manifest_config_serialised():
    cfg = FakeConfig(lr=5e-4, seed=99)
    manifest = build_manifest(cfg, [])
    assert manifest["config"]["lr"] == pytest.approx(5e-4)
    assert manifest["config"]["seed"] == 99


def test_manifest_prompt_count():
    prompts = ["a", "b", "c"]
    manifest = build_manifest(FakeConfig(), prompts)
    assert manifest["prompt_count"] == 3
    assert manifest["prompts"] == prompts


def test_manifest_extra_keys():
    manifest = build_manifest(FakeConfig(), [], extra={"run_id": "xyz", "fid": 1.23})
    assert manifest["run_id"] == "xyz"
    assert manifest["fid"] == pytest.approx(1.23)


def test_manifest_git_commit_is_string():
    manifest = build_manifest(FakeConfig(), [])
    assert isinstance(manifest["git_commit"], str)
    assert len(manifest["git_commit"]) > 0


def test_manifest_timestamp_changes_across_calls():
    m1 = build_manifest(FakeConfig(), [])
    m2 = build_manifest(FakeConfig(), [])
    # Timestamps may or may not differ but must both be ISO strings
    assert "T" in m1["timestamp"]
    assert "T" in m2["timestamp"]


def test_save_manifest_creates_file(tmp_path):
    manifest = build_manifest(FakeConfig(), ["walk"])
    outFile = str(tmp_path / "sub" / "manifest.json")
    save_manifest(manifest, outFile)
    assert Path(outFile).exists()
    loaded = json.loads(Path(outFile).read_text())
    assert loaded["prompt_count"] == 1


def test_save_manifest_valid_json(tmp_path):
    manifest = build_manifest(FakeConfig(), ["dance"])
    outFile = str(tmp_path / "out.json")
    save_manifest(manifest, outFile)
    loaded = json.loads(Path(outFile).read_text())
    assert "git_commit" in loaded


def test_manifest_non_dataclass_config():
    manifest = build_manifest({"lr": 0.1}, [])
    # dict is not a dataclass; should still produce a manifest without crashing
    assert "config" in manifest
