"""Pose-prefix conditioning smoke + behaviour test.

The ``seed_latent`` arg on :meth:`TextToMotionSSM.forward` lets the model
warm-start from a prior action's encoded latents. This test verifies:

  1. The argument is accepted and the model returns the correct output shape
     (the seed portion is dropped, so the output length is unchanged).
  2. Passing a non-zero seed actually CHANGES the output vs the no-seed call
     (proves the seed is plumbed through the SSM, not silently ignored).
  3. The seed length is validated against ``max_motion_length`` and raises
     when the prefix + prediction exceed the model's positional capacity.

We deliberately do NOT compare against the parallel offline forward with
``concat(seed, prediction)`` -- the semantics differ (seed has no text-cond
mixing in the way the prediction does, mirroring a real transition).
"""
from __future__ import annotations

import unittest

import torch

from src.architecture.nn_models import TextToMotionSSM


class TinyCfg:

    def __init__(self) -> None:
        self.d_model = 64
        self.d_state = 16
        self.n_layers = 2
        self.motion_dim = 168
        self.text_embed_dim = 64
        self.max_motion_length = 32  # 8 latent steps with down_t=4
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


class TestSeedLatentForward(unittest.TestCase):

    def test_seed_latent_preserves_output_shape(self):
        torch.manual_seed(0)
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).eval()
        tokens = torch.tensor([[3, 5, 7, 9]], dtype=torch.long)
        # Predict 4 latent steps (= 16 frames) with a 2-step warm-start seed.
        motion_len = 16  # 4 latent steps
        seed = torch.randn(1, 2, cfg.d_model)

        with torch.no_grad():
            logits_no_seed, _ = model(tokens, motion_length=motion_len)
            logits_with_seed, _ = model(tokens, motion_length=motion_len, seed_latent=seed)
        self.assertEqual(logits_no_seed.shape, logits_with_seed.shape)
        # Sanity: shape is (B, latent_len, K, V) where latent_len excludes seed
        self.assertEqual(logits_no_seed.shape[1], motion_len // cfg.rvq_down_t)

    def test_seed_latent_actually_changes_output(self):
        """If the seed is silently ignored, the no-seed and with-seed outputs
        would match. They must NOT match -- the seed advances the SSM state.
        """
        torch.manual_seed(1)
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).eval()
        tokens = torch.tensor([[3, 5, 7, 9]], dtype=torch.long)
        seed = torch.randn(1, 2, cfg.d_model)

        with torch.no_grad():
            logits_no_seed, _ = model(tokens, motion_length=16)
            logits_with_seed, _ = model(tokens, motion_length=16, seed_latent=seed)
        diff = (logits_with_seed - logits_no_seed).abs().max()
        self.assertGreater(
            diff.item(), 1e-3,
            "seed_latent had no effect on the output -- it's being dropped "
            "before the SSM scan, or the residual hidden state isn't reaching "
            "the prediction-step outputs",
        )

    def test_seed_too_long_raises(self):
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).eval()
        tokens = torch.tensor([[3, 5, 7, 9]], dtype=torch.long)
        # max_motion_length=32 / down_t=4 → latent_length=8.
        # Ask for 16-frame prediction (4 latent steps) + 6-step seed = 10 > 8.
        seed = torch.randn(1, 6, cfg.d_model)

        with self.assertRaises(ValueError):
            model(tokens, motion_length=16, seed_latent=seed)


if __name__ == "__main__":
    unittest.main()
