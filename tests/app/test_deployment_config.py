from pathlib import Path

import pytest

from text2motion.app.config import ProjectPathsConfig, load_config


def test_default_deployment_paths_are_portable_atomic_bundles():
    config = load_config("configs/default.yaml")

    assert config.paths.hml3d_out_dir == Path("data/HumanML3D_official")
    assert config.paths.tokenizer_checkpoint.name == "fsq_g8_v1024.pt"
    assert config.paths.generator_checkpoint("transformer").name.endswith("_last.pt")
    assert config.paths.generator_checkpoint("mamba").name.endswith("_last.pt")


def test_deployment_environment_overrides_are_typed(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    tokenizer = tmp_path / "tokenizer.pt"
    monkeypatch.setenv("TEXT2MOTION_DATASET_DIR", str(dataset))
    monkeypatch.setenv("TEXT2MOTION_TOKENIZER_CHECKPOINT", str(tokenizer))
    monkeypatch.setenv("TEXT2MOTION_SERVICE_PORT", "9012")
    monkeypatch.setenv("TEXT2MOTION_IDLE_SECONDS", "45")
    monkeypatch.setenv("TEXT2MOTION_SERVICE_HOST", "legacy.internal")
    monkeypatch.setenv("TEXT2MOTION_CONNECT_HOST", "motion.internal")

    config = load_config("configs/default.yaml")

    assert config.paths.hml3d_out_dir == dataset
    assert config.paths.tokenizer_checkpoint == tokenizer
    assert config.service.port == 9012
    assert config.service.idle_seconds == 45
    assert config.service.connect_host == "motion.internal"


def test_legacy_service_host_remains_a_connect_host_fallback(monkeypatch):
    monkeypatch.setenv("TEXT2MOTION_SERVICE_HOST", "legacy.internal")

    config = load_config("configs/default.yaml")

    assert config.service.connect_host == "legacy.internal"


def test_missing_model_selection_fails_loudly():
    paths = ProjectPathsConfig()

    with pytest.raises(ValueError, match="tokenizer_checkpoint_path"):
        _ = paths.tokenizer_checkpoint
    with pytest.raises(ValueError, match="generator_checkpoint_paths.transformer"):
        paths.generator_checkpoint("transformer")
