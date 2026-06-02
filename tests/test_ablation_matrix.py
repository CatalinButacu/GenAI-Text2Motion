"""Tests for the extended ablation matrix (US-05)."""
from __future__ import annotations

import pytest

from scripts.evaluation.evaluate_ablation import ALL_CONFIGS, pipeline_config_for

EXPECTED_CONFIGS = ["full", "no_m2", "no_film", "no_bidirectional", "no_m1_spacy", "cpu_baseline"]


class TestAllConfigs:
    def test_expected_configs_present(self):
        for name in EXPECTED_CONFIGS:
            assert name in ALL_CONFIGS

    def test_has_six_configs(self):
        assert len(ALL_CONFIGS) == 6


class TestPipelineConfigFor:
    def test_full_default_flags(self, tmp_path):
        cfg = pipeline_config_for("full", str(tmp_path))
        assert cfg.planner.random_layout is False
        assert cfg.motion.use_film is True
        assert cfg.motion.bidirectional is True
        assert cfg.understanding.use_spacy is True

    def test_no_m2_sets_random_layout(self, tmp_path):
        cfg = pipeline_config_for("no_m2", str(tmp_path))
        assert cfg.planner.random_layout is True

    def test_no_film_sets_use_film_false(self, tmp_path):
        cfg = pipeline_config_for("no_film", str(tmp_path))
        assert cfg.motion.use_film is False
        assert cfg.planner.random_layout is False

    def test_no_bidirectional_sets_flag(self, tmp_path):
        cfg = pipeline_config_for("no_bidirectional", str(tmp_path))
        assert cfg.motion.bidirectional is False

    def test_no_m1_spacy_disables_spacy(self, tmp_path):
        cfg = pipeline_config_for("no_m1_spacy", str(tmp_path))
        assert cfg.understanding.use_spacy is False

    def test_cpu_baseline_forces_cpu(self, tmp_path):
        cfg = pipeline_config_for("cpu_baseline", str(tmp_path), device="cuda")
        assert cfg.device == "cpu"

    def test_unknown_config_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown config"):
            pipeline_config_for("nonexistent_config", str(tmp_path))

    def test_all_configs_build_without_error(self, tmp_path):
        for name in ALL_CONFIGS:
            cfg = pipeline_config_for(name, str(tmp_path))
            assert cfg is not None
