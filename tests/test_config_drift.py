"""Drift catcher: src/shared/constants.py must agree with configs/motion_ssm.yaml.

Both files define the cloud-headline architecture. If one drifts away from the
other, checkpoints trained under one will fail to load under code that defaults
to the other. This test fires loud the moment they diverge.

The YAML is the source of truth — if this test fails after a deliberate
architecture change, update constants.py to match the YAML, not the other way
round.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.shared.config import SSM, TrainingConfig, load_yaml_config
from src.shared.constants import SMPLX

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADLINE_YAML = REPO_ROOT / "configs" / "motion_ssm.yaml"


class TestConfigDrift(unittest.TestCase):

    def test_yaml_exists(self):
        self.assertTrue(
            HEADLINE_YAML.exists(),
            f"missing source-of-truth config at {HEADLINE_YAML}",
        )

    def test_constants_match_yaml(self):
        flat = load_yaml_config(HEADLINE_YAML)
        self.assertEqual(
            SSM.d_model, flat["d_model"],
            "src/shared/nn_config.SSM.d_model != YAML architecture.d_model",
        )
        self.assertEqual(
            SSM.d_state, flat["d_state"],
            "src/shared/nn_config.SSM.d_state != YAML architecture.d_state",
        )
        self.assertEqual(
            SSM.n_layers, flat["n_layers"],
            "src/shared/nn_config.SSM.n_layers != YAML architecture.n_layers",
        )
        self.assertEqual(
            SMPLX.pose_dim, flat["motion_dim"],
            "src/shared/constants.SMPLX.pose_dim != YAML architecture.motion_dim",
        )

    def test_training_config_from_yaml_round_trip(self):
        """from_yaml() returns a config whose model fields match the YAML exactly."""
        cfg = TrainingConfig.from_yaml(HEADLINE_YAML)
        flat = load_yaml_config(HEADLINE_YAML)
        # Spot-check a few load-bearing fields across all three YAML sections.
        self.assertEqual(cfg.d_model, flat["d_model"])
        self.assertEqual(cfg.n_layers, flat["n_layers"])
        self.assertEqual(cfg.rvq_n_codebooks, flat["rvq_n_codebooks"])
        self.assertEqual(cfg.batch_size, flat["batch_size"])
        self.assertEqual(cfg.num_epochs, flat["num_epochs"])


if __name__ == "__main__":
    unittest.main()
