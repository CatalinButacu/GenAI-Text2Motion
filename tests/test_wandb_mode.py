"""Regression test for the wandb online/offline mode resolution.

Bug fixed 2026-05-22: init_wandb used to unconditionally setdefault
WANDB_MODE to 'offline' which forced offline mode even when the cloud
startup script had exported WANDB_API_KEY for an online run. Online
runs were silently downgraded.

These tests don't actually call wandb (we don't want network in unit
tests) -- they only verify the env-var resolution logic that
init_wandb applies before delegating to wandb.init.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch


@dataclass
class FakeConfig:
    lr: float = 0.001
    epochs: int = 1


class TestWandbModeResolution(unittest.TestCase):

    def setUp(self) -> None:
        # Snapshot the environ so each test starts clean.
        self.saved = dict(os.environ)
        for k in ("WANDB_MODE", "WANDB_API_KEY", "WANDB_DIR", "WANDB_SILENT"):
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.saved)

    def call_init_wandb_with_no_real_call(self, **kwargs) -> None:
        """Invoke init_wandb but stub wandb.init so we don't hit the network."""
        from src.shared import run_ctx
        with patch.object(run_ctx, "wandb") as m_wandb:
            m_wandb.init = MagicMock(return_value=MagicMock())
            run_ctx.init_wandb(**kwargs)

    def args(self, tmpdir) -> dict:
        return {
            "project": "test", "run_id": "r0",
            "run_dir": tmpdir, "config": FakeConfig(),
        }

    def test_no_api_key_defaults_to_offline(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            self.call_init_wandb_with_no_real_call(**self.args(Path(td)))
            self.assertEqual(os.environ.get("WANDB_MODE"), "offline")

    def test_api_key_present_does_not_force_offline(self):
        """The bug: setdefault made online runs offline. Guard against regression."""
        import tempfile
        from pathlib import Path
        os.environ["WANDB_API_KEY"] = "fake-key-for-test-not-sent"
        with tempfile.TemporaryDirectory() as td:
            self.call_init_wandb_with_no_real_call(**self.args(Path(td)))
            # WANDB_MODE must NOT be forced to "offline" when an API key
            # is present -- wandb's own default (online) should take over.
            self.assertNotEqual(os.environ.get("WANDB_MODE"), "offline")

    def test_explicit_mode_offline_wins_even_with_api_key(self):
        """User can still force offline by setting WANDB_MODE explicitly."""
        import tempfile
        from pathlib import Path
        os.environ["WANDB_API_KEY"] = "fake-key-for-test-not-sent"
        os.environ["WANDB_MODE"] = "offline"
        with tempfile.TemporaryDirectory() as td:
            self.call_init_wandb_with_no_real_call(**self.args(Path(td)))
            self.assertEqual(os.environ.get("WANDB_MODE"), "offline")

    def test_wandb_dir_and_silent_set(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            self.call_init_wandb_with_no_real_call(**self.args(Path(td)))
            self.assertEqual(os.environ.get("WANDB_DIR"), str(Path(td)))
            self.assertEqual(os.environ.get("WANDB_SILENT"), "true")


if __name__ == "__main__":
    unittest.main()
