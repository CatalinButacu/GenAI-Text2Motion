"""Checkpoint round-trip + key-compatibility tests.

Targets the bug class that cost real cloud money on 2026-05-13: SBERT key
mismatch on warm-start silently loaded only part of the model and training
proceeded with random weights for the rest. The contract these tests
enforce:

  1. save_ckpt -> torch.load -> load_compatible recovers identical weights.
  2. load_compatible logs a warning AND skips keys that don't match shape,
     instead of crashing or silently corrupting the model.
  3. load_compatible returns False (and warns) when nothing matches —
     never silently leaves the model on random init.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.modules.motion.training.trainer_utils import (
    find_latest_ckpt,
    load_compatible,
    resolve_ckpt_path,
    restore_checkpoint,
    save_ckpt,
)


def make_tiny_model(d_in: int = 8, d_out: int = 4) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, 16), nn.ReLU(), nn.Linear(16, d_out))


class TestCheckpointRoundTrip(unittest.TestCase):

    def test_save_then_load_recovers_state_exactly(self):
        torch.manual_seed(0)
        source = make_tiny_model()

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ckpt.pt"
            save_ckpt(str(ckpt), {"model_state_dict": source.state_dict()})

            target = make_tiny_model()  # random init
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            ok = load_compatible(target, payload["model_state_dict"], "target")

            self.assertTrue(ok)
            for k_src, k_tgt in zip(source.state_dict(), target.state_dict(), strict=True):
                self.assertTrue(
                    torch.equal(source.state_dict()[k_src], target.state_dict()[k_tgt]),
                    f"weights diverged at {k_src}",
                )


class TestLoadCompatible(unittest.TestCase):
    """The defensive loader is the thin layer between "checkpoint compatible"
    and "checkpoint silently corrupts training" — guard its contract."""

    def test_shape_mismatch_is_skipped_not_crashed(self):
        source = make_tiny_model(d_in=8, d_out=4)
        target = make_tiny_model(d_in=16, d_out=4)  # different first-layer shape

        ok = load_compatible(target, source.state_dict(), "target")
        # Some keys (bias of second layer, etc.) still match — should still report ok=True
        self.assertTrue(ok)
        # The first-layer weight must NOT have been copied (shape mismatch)
        self.assertNotEqual(
            target[0].weight.shape, source[0].weight.shape,
            "shape sanity check failed",
        )

    def test_completely_incompatible_returns_false(self):
        source = make_tiny_model()
        # Build a model with completely different parameter names + shapes
        target = nn.Sequential(nn.Conv1d(3, 8, kernel_size=3))

        ok = load_compatible(target, source.state_dict(), "target")
        self.assertFalse(ok)

    def test_extra_keys_in_checkpoint_are_skipped(self):
        source = make_tiny_model()
        target = make_tiny_model()

        payload = dict(source.state_dict())
        payload["bogus_extra_key"] = torch.zeros(8)

        ok = load_compatible(target, payload, "target")
        self.assertTrue(ok)
        # Target should still have only its original keys, not the bogus one
        self.assertNotIn("bogus_extra_key", target.state_dict())

    def test_partial_match_loads_what_it_can(self):
        """Regression guard for SBERT key mismatch: half the model loads
        from checkpoint, the other half (e.g. the SBERT encoder branch) is
        absent. load_compatible must NOT crash and must load what it can."""
        source = make_tiny_model()
        target = make_tiny_model()

        # Drop one key from the source -> simulate a checkpoint missing weights
        partial = dict(source.state_dict())
        dropped_key = next(iter(partial))
        dropped_value = partial.pop(dropped_key)

        ok = load_compatible(target, partial, "target")
        self.assertTrue(ok)
        # The dropped key in target keeps its random init (not zero, not source's)
        target_val = target.state_dict()[dropped_key]
        self.assertFalse(
            torch.equal(target_val, dropped_value),
            "target unexpectedly received the dropped key's value",
        )


class FakeDataset:
    """Minimal stand-in for a MotionDataset — only `vocab` matters here."""
    def __init__(self, vocab: dict) -> None:
        self.vocab = vocab


class FakeTrainer:
    """Minimal stand-in exposing the attributes restore_checkpoint touches."""

    def __init__(self, model, vocab: dict) -> None:
        self.model = model
        self.device = torch.device("cpu")
        self.train_ds = FakeDataset(vocab=vocab)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")


class TestRestoreCheckpointVocab(unittest.TestCase):
    """Regression guard for the 2026-05 audit: restore_checkpoint used to
    check `hasattr(trainer, "train_ds")` (snake_case) against a camelCase
    `train_ds` attribute, so vocab was silently never restored on warm-start.
    """

    def test_vocab_is_restored_from_checkpoint_on_warm_start(self):
        torch.manual_seed(0)
        model = make_tiny_model()

        saved_vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "walk": 4, "run": 5}
        trainer = FakeTrainer(model, vocab={"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3})

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ck.pt"
            save_ckpt(str(ckpt), {
                "model_state_dict": model.state_dict(),
                "vocab": saved_vocab,
                "val_loss": 0.42,
            })
            restore_checkpoint(trainer, str(ckpt), warm_start=True)

        self.assertEqual(trainer.train_ds.vocab, saved_vocab,
                         "vocab from checkpoint was not restored onto trainer.train_ds")
        self.assertEqual(trainer.best_loss, 0.42)

    def test_vocab_is_restored_on_full_resume(self):
        model = make_tiny_model()
        saved_vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "jump": 4}
        trainer = FakeTrainer(model, vocab={"<PAD>": 0})

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ck.pt"
            save_ckpt(str(ckpt), {
                "model_state_dict": model.state_dict(),
                "vocab": saved_vocab,
                "val_loss": 0.5,
                "epoch": 7,
                "global_step": 1234,
            })
            restore_checkpoint(trainer, str(ckpt), warm_start=False)

        self.assertEqual(trainer.train_ds.vocab, saved_vocab)
        self.assertEqual(trainer.start_epoch, 8)
        self.assertEqual(trainer.step, 1234)

    def test_no_vocab_key_leaves_trainer_vocab_untouched(self):
        model = make_tiny_model()
        original = {"<PAD>": 0, "existing": 99}
        trainer = FakeTrainer(model, vocab=original)

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ck.pt"
            save_ckpt(str(ckpt), {"model_state_dict": model.state_dict()})  # no "vocab" key
            restore_checkpoint(trainer, str(ckpt), warm_start=True)

        self.assertEqual(trainer.train_ds.vocab, original)


class TestCheckpointPathHelpers(unittest.TestCase):

    def test_find_latest_picks_highest_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            for epoch in (1, 5, 12, 3, 9):
                (Path(td) / f"checkpoint_epoch{epoch}.pt").write_bytes(b"x")
            latest = find_latest_ckpt(td)
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertTrue(latest.endswith("checkpoint_epoch12.pt"))

    def test_find_latest_in_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(find_latest_ckpt(td))

    def test_resolve_ckpt_finds_explicit_file_in_dir(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "checkpoint_epoch3.pt"
            target.write_bytes(b"x")
            resolved = resolve_ckpt_path("checkpoint_epoch3.pt", td)
            self.assertEqual(resolved, str(target))

    def test_resolve_ckpt_latest_with_no_ckpts_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(resolve_ckpt_path("latest", td))


if __name__ == "__main__":
    unittest.main()
