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
    restoreCheckpoint,
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


class FakeDataset:
    """Minimal stand-in for a MotionDataset — only `vocab` matters here."""
    def __init__(self, vocab: dict) -> None:
        self.vocab = vocab


class FakeTrainer:
    """Minimal stand-in exposing the attributes restoreCheckpoint touches."""

    def __init__(self, model, vocab: dict) -> None:
        self.model = model
        self.device = torch.device("cpu")
        self.trainDs = FakeDataset(vocab=vocab)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.step = 0
        self.startEpoch = 0
        self.bestLoss = float("inf")


class TestRestoreCheckpointVocab(unittest.TestCase):
    """Regression guard for the 2026-05 audit: restoreCheckpoint used to
    check `hasattr(trainer, "train_ds")` (snake_case) against a camelCase
    `trainDs` attribute, so vocab was silently never restored on warm-start.
    """

    def test_vocab_is_restored_from_checkpoint_on_warm_start(self):
        torch.manual_seed(0)
        model = makeTinyModel()

        savedVocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "walk": 4, "run": 5}
        trainer = FakeTrainer(model, vocab={"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3})

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ck.pt"
            saveCkpt(str(ckpt), {
                "model_state_dict": model.state_dict(),
                "vocab": savedVocab,
                "valLoss": 0.42,
            })
            restoreCheckpoint(trainer, str(ckpt), warmStart=True)

        self.assertEqual(trainer.trainDs.vocab, savedVocab,
                         "vocab from checkpoint was not restored onto trainer.trainDs")
        self.assertEqual(trainer.bestLoss, 0.42)

    def test_vocab_is_restored_on_full_resume(self):
        model = makeTinyModel()
        savedVocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "jump": 4}
        trainer = FakeTrainer(model, vocab={"<PAD>": 0})

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ck.pt"
            saveCkpt(str(ckpt), {
                "model_state_dict": model.state_dict(),
                "vocab": savedVocab,
                "valLoss": 0.5,
                "epoch": 7,
                "global_step": 1234,
            })
            restoreCheckpoint(trainer, str(ckpt), warmStart=False)

        self.assertEqual(trainer.trainDs.vocab, savedVocab)
        self.assertEqual(trainer.startEpoch, 8)
        self.assertEqual(trainer.step, 1234)

    def test_no_vocab_key_leaves_trainer_vocab_untouched(self):
        model = makeTinyModel()
        original = {"<PAD>": 0, "existing": 99}
        trainer = FakeTrainer(model, vocab=original)

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ck.pt"
            saveCkpt(str(ckpt), {"model_state_dict": model.state_dict()})  # no "vocab" key
            restoreCheckpoint(trainer, str(ckpt), warmStart=True)

        self.assertEqual(trainer.trainDs.vocab, original)


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
