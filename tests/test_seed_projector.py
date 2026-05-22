"""seed_from_latent must lift tokenizer latents to d_model and round-trip
through forward() as a valid pose-prefix.

This test is the unit-level proof that the trainer-side curriculum has a
clean integration point: the model exposes one method that converts
``(B, P, rvq_latent_dim)`` -- the natural output of ``tokenizer.rvq.decode``
-- into the ``(B, P, d_model)`` seed_latent that :meth:`forward` already
accepts.

If this test passes:
  - the trainer can call ``model.seed_from_latent(prefix_latent)`` to get
    a valid seed for ``forward(... seed_latent=seed)``
  - the projection layer is exercised so it actually has gradients
  - the existing forward(seed_latent=...) plumbing still works through
    the projected tensor
"""
from __future__ import annotations

import unittest

import torch

from src.modules.motion.nn_models import TextToMotionSSM


class TinyCfg:

    def __init__(self) -> None:
        self.d_model = 64
        self.d_state = 16
        self.n_layers = 2
        self.motion_dim = 168
        self.text_embed_dim = 64
        self.max_motion_length = 32
        self.max_text_length = 8
        self.vocab_size = 32
        self.bidirectional = False
        self.use_film = True
        self.use_sbert = False
        self.sbert_model = "all-MiniLM-L6-v2"
        self.freeze_sbert = True
        self.gradient_checkpointing = False
        self.arch = "independent"
        self.rvq_latent_dim = 16
        self.rvq_n_codebooks = 2
        self.rvq_codebook_size = 8
        self.rvq_down_t = 4


class TestSeedProjector(unittest.TestCase):

    def test_projects_to_d_model(self):
        torch.manual_seed(0)
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).eval()
        # Tokenizer-shaped latent: (B, P, rvq_latent_dim)
        prefix_latent = torch.randn(2, 3, cfg.rvq_latent_dim)
        seed = model.seed_from_latent(prefix_latent)
        self.assertEqual(seed.shape, (2, 3, cfg.d_model))

    def test_round_trips_through_forward(self):
        """Project tokenizer latent → use as seed → forward returns valid logits."""
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).eval()
        tokens = torch.tensor([[5, 7, 9, 11]], dtype=torch.long)
        prefix_latent = torch.randn(1, 2, cfg.rvq_latent_dim)

        with torch.no_grad():
            seed = model.seed_from_latent(prefix_latent)
            logits, length_pred = model(tokens, motion_length=16, seed_latent=seed)
        self.assertEqual(
            logits.shape, (1, 16 // cfg.rvq_down_t, cfg.rvq_n_codebooks, cfg.rvq_codebook_size),
        )
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(length_pred).all())

    def test_projector_has_gradient(self):
        """End-to-end gradient through the projector confirms it's part of the
        trainable param set, so the trainer-side curriculum will actually
        update it on backprop.
        """
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).train()
        tokens = torch.tensor([[5, 7, 9, 11]], dtype=torch.long)
        prefix_latent = torch.randn(1, 2, cfg.rvq_latent_dim, requires_grad=False)
        seed = model.seed_from_latent(prefix_latent)
        logits, _ = model(tokens, motion_length=16, seed_latent=seed)
        # any scalar loss
        logits.sum().backward()
        grad = model.seed_projector.weight.grad
        self.assertIsNotNone(grad)
        assert grad is not None  # type narrowing for pyright
        self.assertGreater(
            grad.abs().sum().item(), 0.0,
            "seed_projector saw no gradient -- the projection is being "
            "bypassed somewhere upstream of forward()",
        )


if __name__ == "__main__":
    unittest.main()
