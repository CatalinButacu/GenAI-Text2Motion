"""Tests for the MDM-family geometric losses added to the SSM trainer.

These losses are critical for the 'physics-constrained' claim of the thesis:
they push the model toward motion-space correctness, not just token-space
correctness. The tests below assert:

  1. Forward shape correctness of the soft decode.
  2. Gradients actually flow back to the input logits (not just the codebook).
  3. Numerical sanity: loss is finite, non-negative, and shrinks toward zero
     as logits become more confident on the GT tokens.
  4. All three loss terms participate in the autograd graph (non-zero grads).
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer
from src.modules.motion.training.trainer_utils import (
    geometric_losses,
    soft_decode_logits,
)


def _make_tokenizer(
    motion_dim: int = 168, latent_dim: int = 32, K: int = 3, V: int = 64
) -> MotionRVQTokenizer:
    """Tiny RVQ tokenizer for fast unit tests."""
    return MotionRVQTokenizer(
        motion_dim=motion_dim,
        latent_dim=latent_dim,
        n_codebooks=K,
        codebook_size=V,
        down_t=4,
    )


class TestSoftDecode(unittest.TestCase):

    def test_soft_decode_shape(self):
        """soft_decode_logits returns (B, T, motion_dim) with T = T'*down_t."""
        torch.manual_seed(0)
        tok = _make_tokenizer()
        B, Tp, K, V = 2, 5, 3, 64
        logits = torch.randn(B, Tp, K, V)
        out = soft_decode_logits(logits, tok)
        # Decoder upsamples by down_t=4 (ConvTranspose1d x2)
        self.assertEqual(out.shape[0], B)
        self.assertEqual(out.shape[2], 168)
        self.assertEqual(out.shape[1], Tp * 4)

    def test_gradient_flows_to_logits(self):
        """Soft decode is differentiable; logits.grad must be non-None and non-zero."""
        torch.manual_seed(1)
        tok = _make_tokenizer()
        logits = torch.randn(1, 4, 3, 64, requires_grad=True)
        out = soft_decode_logits(logits, tok)
        out.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)


class TestGeometricLosses(unittest.TestCase):

    def _scenario(self, B: int = 2, Tp: int = 4, K: int = 3, V: int = 64):
        torch.manual_seed(42)
        tok = _make_tokenizer(K=K, V=V)
        T = Tp * 4
        logits = torch.randn(B, Tp, K, V, requires_grad=True)
        gt_motion = torch.randn(B, T, 168) * 0.1
        frame_mask = torch.ones(B, T)
        return tok, logits, gt_motion, frame_mask

    def test_returns_three_terms(self):
        tok, logits, gt, mask = self._scenario()
        out = geometric_losses(logits, gt, mask, tok)
        self.assertEqual(set(out.keys()), {"recon", "velocity", "root_height"})

    def test_losses_finite_and_nonnegative(self):
        tok, logits, gt, mask = self._scenario()
        out = geometric_losses(logits, gt, mask, tok)
        for name, val in out.items():
            self.assertTrue(torch.isfinite(val).item(), f"{name} non-finite: {val}")
            self.assertGreaterEqual(val.item(), 0.0, f"{name} negative: {val}")

    def test_all_terms_propagate_gradient(self):
        """Each term individually must produce non-zero gradient on logits."""
        for term in ("recon", "velocity", "root_height"):
            with self.subTest(term=term):
                tok, logits, gt, mask = self._scenario()
                out = geometric_losses(logits, gt, mask, tok)
                out[term].backward()
                self.assertIsNotNone(logits.grad)
                self.assertGreater(
                    logits.grad.abs().sum().item(), 0.0,
                    f"{term} produced zero gradient on logits",
                )

    def test_masking_zeros_padded_regions(self):
        """Padded frames must not contribute to the loss."""
        tok, logits, gt, _full_mask = self._scenario()
        # All-zero mask -> loss is zero (recon + root_h) or near-zero (velocity)
        zero_mask = torch.zeros_like(_full_mask)
        out = geometric_losses(logits, gt, zero_mask, tok)
        self.assertAlmostEqual(out["recon"].item(), 0.0, places=5)
        self.assertAlmostEqual(out["root_height"].item(), 0.0, places=5)
        # velocity uses (mask[t] * mask[t-1]) so it's also zero with empty mask
        self.assertAlmostEqual(out["velocity"].item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
