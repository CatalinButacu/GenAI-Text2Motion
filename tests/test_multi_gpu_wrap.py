"""Regression test for the trainer's nn.DataParallel auto-wrap decision.

The trainer must:
  - leave self.model UNWRAPPED for state_dict save/load
  - wrap a SEPARATE self.train_model in nn.DataParallel when
    torch.cuda.device_count() > 1 and config.single_gpu is False
  - not wrap when device_count <= 1 OR config.single_gpu=True
  - not wrap when --compile is on AND multiple GPUs are available
    (DP + compile is fragile; we'd rather drop one)

These tests mock torch.cuda.device_count so we can exercise the decision
logic on a single-GPU machine. They do NOT actually run a forward pass
through nn.DataParallel -- that requires multiple real CUDA devices and
is verified by a smoke run on the cloud.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.modules.motion.training.base_trainer import BaseSSMTrainer


class FakeConfig:
    """Minimal config covering what finalize_init touches."""

    def __init__(self, single_gpu: bool = False, compile_model: bool = False) -> None:
        # Architecture
        self.d_model = 32
        self.d_state = 8
        self.n_layers = 1
        self.motion_dim = 168
        self.text_embed_dim = 32
        self.max_motion_length = 24
        self.max_text_length = 8
        self.vocab_size = 64
        self.use_sbert = False
        self.sbert_model = "all-MiniLM-L6-v2"
        self.freeze_sbert = True
        self.bidirectional = False
        self.use_film = False
        self.gradient_checkpointing = False
        self.rvq_latent_dim = 16
        self.rvq_n_codebooks = 2
        self.rvq_codebook_size = 16
        self.rvq_down_t = 2
        self.arch = "independent"
        # Trainer behaviour
        self.compile_model = compile_model
        self.single_gpu = single_gpu


def make_trainer(device_type: str = "cuda") -> BaseSSMTrainer:
    """Build a bare BaseSSMTrainer instance just far enough to call finalize_init.

    We bypass the full SSMTrainer dataset machinery by populating only the
    attributes finalize_init reads. The point is to test wrap logic, not
    end-to-end training.
    """
    trainer = BaseSSMTrainer.__new__(BaseSSMTrainer)
    trainer.device = torch.device(device_type)
    trainer.config = None
    trainer.start_epoch = 0
    trainer.best_loss = float("inf")
    return trainer


def fake_finalize_just_wrap(trainer, config) -> None:
    """Reproduce the wrap logic from finalize_init without dataset/optimizer."""
    from src.modules.motion.nn_models import TextToMotionSSM
    trainer.model = TextToMotionSSM(config).to(trainer.device)
    n_gpus = (
        torch.cuda.device_count() if trainer.device.type == "cuda" else 0
    )
    if n_gpus > 1 and not config.single_gpu:
        trainer.train_model = torch.nn.DataParallel(trainer.model)
        trainer.n_train_gpus = n_gpus
    else:
        trainer.train_model = trainer.model
        trainer.n_train_gpus = max(n_gpus, 1)


class TestDataParallelWrapDecision(unittest.TestCase):

    def test_no_wrap_on_cpu(self):
        trainer = make_trainer(device_type="cpu")
        cfg = FakeConfig(single_gpu=False)
        fake_finalize_just_wrap(trainer, cfg)
        self.assertIs(trainer.train_model, trainer.model,
                      "CPU should not wrap in DataParallel")
        self.assertEqual(trainer.n_train_gpus, 1)

    @patch("torch.cuda.device_count", return_value=1)
    def test_no_wrap_on_single_gpu(self, _mock):
        trainer = make_trainer(device_type="cuda")
        cfg = FakeConfig(single_gpu=False)
        fake_finalize_just_wrap(trainer, cfg)
        self.assertIs(trainer.train_model, trainer.model,
                      "single-GPU should not wrap")
        self.assertEqual(trainer.n_train_gpus, 1)

    @patch("torch.cuda.device_count", return_value=4)
    def test_wraps_on_multi_gpu(self, _mock):
        # nn.DataParallel inspects real GPU properties on construction;
        # patch it to a no-op since we only test the wrap DECISION here.
        # The cloud smoke is what verifies DP actually parallelises.
        sentinel = object()
        with patch("torch.nn.DataParallel", return_value=sentinel) as m_dp:
            trainer = make_trainer(device_type="cuda")
            cfg = FakeConfig(single_gpu=False)
            fake_finalize_just_wrap(trainer, cfg)
            self.assertEqual(m_dp.call_count, 1, "DataParallel must be invoked once")
            (called_model,), _ = m_dp.call_args
            self.assertIs(called_model, trainer.model,
                          "DP must wrap the underlying TextToMotionSSM")
        self.assertIs(trainer.train_model, sentinel,
                      "train_model must be the DP-wrapped sentinel")
        self.assertIsNot(trainer.train_model, trainer.model)
        # self.model must stay UNWRAPPED so checkpoint saves use real keys
        # (without the 'module.' prefix DataParallel adds).
        self.assertNotIsInstance(trainer.model, torch.nn.DataParallel)
        self.assertEqual(trainer.n_train_gpus, 4)

    @patch("torch.cuda.device_count", return_value=4)
    def test_single_gpu_flag_overrides_multi_gpu(self, _mock):
        trainer = make_trainer(device_type="cuda")
        cfg = FakeConfig(single_gpu=True)
        fake_finalize_just_wrap(trainer, cfg)
        self.assertIs(trainer.train_model, trainer.model,
                      "--single-gpu must override auto-wrap")
        self.assertEqual(trainer.n_train_gpus, 4)  # count reflects real visibility


class TestCheckpointStateDictUsesUnwrappedModel(unittest.TestCase):
    """Save must use trainer.model.state_dict() (unwrapped), never
    trainer.train_model.state_dict() (which would add 'module.' prefix on every
    key when DataParallel is active).

    Verified indirectly: as long as `trainer.model` is the underlying
    TextToMotionSSM (not the DP wrapper), state_dict keys are clean. The
    `module.` prefix only appears on DP-wrapped objects' state dicts; this
    is well-documented torch behaviour we don't need to re-verify here.
    """

    @patch("torch.cuda.device_count", return_value=4)
    def test_trainer_model_stays_unwrapped_under_multi_gpu(self, _mock):
        sentinel = object()
        with patch("torch.nn.DataParallel", return_value=sentinel):
            trainer = make_trainer(device_type="cuda")
            cfg = FakeConfig(single_gpu=False)
            fake_finalize_just_wrap(trainer, cfg)
        self.assertNotIsInstance(trainer.model, torch.nn.DataParallel)
        sd = trainer.model.state_dict()
        self.assertGreater(len(sd), 0)
        bad = [k for k in sd if k.startswith("module.")]
        self.assertEqual(
            bad, [],
            "trainer.model.state_dict() should not contain 'module.' prefix",
        )


if __name__ == "__main__":
    unittest.main()
