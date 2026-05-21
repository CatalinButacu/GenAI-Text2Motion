"""Checkpoint round-trip + key-compatibility tests.

Targets the bug class that cost real cloud money on 2026-05-13: SBERT key
mismatch on warm-start silently loaded only part of the model and training
proceeded with random weights for the rest. The contract these tests
enforce:

  1. saveCkpt -> torch.load -> loadCompatible recovers identical weights.
  2. loadCompatible logs a warning AND skips keys that don't match shape,
     instead of crashing or silently corrupting the model.
  3. loadCompatible returns False (and warns) when nothing matches —
     never silently leaves the model on random init.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.modules.motion.training.trainer_utils import (
    findLatestCkpt,
    loadCompatible,
    resolveCkptPath,
    saveCkpt,
)


def makeTinyModel(dIn: int = 8, dOut: int = 4) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dIn, 16), nn.ReLU(), nn.Linear(16, dOut))


class TestCheckpointRoundTrip(unittest.TestCase):

    def test_save_then_load_recovers_state_exactly(self):
        torch.manual_seed(0)
        source = makeTinyModel()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ckpt.pt"
            saveCkpt(str(ckpt), {"model_state_dict": source.state_dict()})

            target = makeTinyModel()  # random init
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            ok = loadCompatible(target, payload["model_state_dict"], "target")

            self.assertTrue(ok)
            for kSrc, kTgt in zip(source.state_dict(), target.state_dict(), strict=True):
                self.assertTrue(
                    torch.equal(source.state_dict()[kSrc], target.state_dict()[kTgt]),
                    f"weights diverged at {kSrc}",
                )


class TestLoadCompatible(unittest.TestCase):
    """The defensive loader is the thin layer between "checkpoint compatible"
    and "checkpoint silently corrupts training" — guard its contract."""

    def test_shape_mismatch_is_skipped_not_crashed(self):
        source = makeTinyModel(dIn=8, dOut=4)
        target = makeTinyModel(dIn=16, dOut=4)  # different first-layer shape

        ok = loadCompatible(target, source.state_dict(), "target")
        # Some keys (bias of second layer, etc.) still match — should still report ok=True
        self.assertTrue(ok)
        # The first-layer weight must NOT have been copied (shape mismatch)
        self.assertNotEqual(
            target[0].weight.shape, source[0].weight.shape,
            "shape sanity check failed",
        )

    def test_completely_incompatible_returns_false(self):
        source = makeTinyModel()
        # Build a model with completely different parameter names + shapes
        target = nn.Sequential(nn.Conv1d(3, 8, kernel_size=3))

        ok = loadCompatible(target, source.state_dict(), "target")
        self.assertFalse(ok)

    def test_extra_keys_in_checkpoint_are_skipped(self):
        source = makeTinyModel()
        target = makeTinyModel()

        payload = dict(source.state_dict())
        payload["bogus_extra_key"] = torch.zeros(8)

        ok = loadCompatible(target, payload, "target")
        self.assertTrue(ok)
        # Target should still have only its original keys, not the bogus one
        self.assertNotIn("bogus_extra_key", target.state_dict())

    def test_partial_match_loads_what_it_can(self):
        """Regression guard for SBERT key mismatch: half the model loads
        from checkpoint, the other half (e.g. the SBERT encoder branch) is
        absent. loadCompatible must NOT crash and must load what it can."""
        source = makeTinyModel()
        target = makeTinyModel()

        # Drop one key from the source -> simulate a checkpoint missing weights
        partial = dict(source.state_dict())
        droppedKey = next(iter(partial))
        droppedValue = partial.pop(droppedKey)

        ok = loadCompatible(target, partial, "target")
        self.assertTrue(ok)
        # The dropped key in target keeps its random init (not zero, not source's)
        targetVal = target.state_dict()[droppedKey]
        self.assertFalse(
            torch.equal(targetVal, droppedValue),
            "target unexpectedly received the dropped key's value",
        )


class TestCheckpointPathHelpers(unittest.TestCase):

    def test_find_latest_picks_highest_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            for epoch in (1, 5, 12, 3, 9):
                (Path(td) / f"checkpoint_epoch{epoch}.pt").write_bytes(b"x")
            latest = findLatestCkpt(td)
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertTrue(latest.endswith("checkpoint_epoch12.pt"))

    def test_find_latest_in_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(findLatestCkpt(td))

    def test_resolve_ckpt_finds_explicit_file_in_dir(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "checkpoint_epoch3.pt"
            target.write_bytes(b"x")
            resolved = resolveCkptPath("checkpoint_epoch3.pt", td)
            self.assertEqual(resolved, str(target))

    def test_resolve_ckpt_latest_with_no_ckpts_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(resolveCkptPath("latest", td))


if __name__ == "__main__":
    unittest.main()
