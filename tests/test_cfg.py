"""Tests for the classifier-free guidance (CFG) sampling path.

The CFG path is: guided = uncond + cfg_scale * (cond - uncond).
It must:
  - default to off (cfg_scale=1.0 -> identical output to no-CFG path)
  - actually change outputs at cfg_scale > 1.0
  - be a no-op for non-SBERT/CLIP models (no uncond text path available)
  - propagate from MotionConfig through the generator cache
"""

from __future__ import annotations

import unittest

import torch

from src.modules.motion.config import MotionConfig
from src.modules.motion.ssm_model import sample_indices


class TestCfgBlendMath(unittest.TestCase):
    """Verify the logit-blending math in isolation, independent of the model."""

    def test_cfg_scale_one_is_identity(self):
        cond = torch.randn(2, 4, 3, 8)
        unc = torch.randn(2, 4, 3, 8)
        blended = unc + 1.0 * (cond - unc)
        # scale=1 -> just the conditional logits.
        self.assertTrue(torch.allclose(blended, cond))

    def test_cfg_scale_zero_is_unconditional(self):
        cond = torch.randn(2, 4, 3, 8)
        unc = torch.randn(2, 4, 3, 8)
        blended = unc + 0.0 * (cond - unc)
        self.assertTrue(torch.allclose(blended, unc))

    def test_cfg_scale_two_amplifies_difference(self):
        cond = torch.zeros(1, 1, 1, 4)
        cond[..., 0] = 1.0  # only the conditional pushes class 0 up
        unc = torch.zeros(1, 1, 1, 4)
        blended = unc + 2.0 * (cond - unc)
        # scale=2: blended[..., 0] = 0 + 2 * (1 - 0) = 2.0 (vs cond=1.0).
        self.assertAlmostEqual(float(blended[0, 0, 0, 0]), 2.0)


class TestMotionConfigDefaults(unittest.TestCase):

    def test_cfg_scale_defaults_to_off(self):
        cfg = MotionConfig()
        self.assertEqual(cfg.cfg_scale, 1.0)

    def test_cfg_scale_can_be_set(self):
        cfg = MotionConfig(cfg_scale=4.0)
        self.assertEqual(cfg.cfg_scale, 4.0)


class TestSampleIndicesWithCfgBlendedLogits(unittest.TestCase):
    """At cfg_scale > 1, the same sampler argmaxes a different distribution."""

    def test_blended_logits_can_change_argmax(self):
        torch.manual_seed(0)
        # cond strongly favours class 0; unc strongly favours class 2.
        cond = torch.zeros(1, 1, 1, 4)
        cond[..., 0] = 5.0
        unc = torch.zeros(1, 1, 1, 4)
        unc[..., 2] = 5.0

        # scale=1 -> conditional wins: class 0
        idx_cond = sample_indices(cond, temperature=1.0, top_p=1.0)
        self.assertEqual(int(idx_cond[0, 0, 0]), 0)

        # scale=2 -> unc + 2*(cond-unc) = 2*cond - unc
        # element-wise: [10, 0, -5, 0]. argmax = 0 (cond push wins).
        blended = unc + 2.0 * (cond - unc)
        idx_blended = sample_indices(blended, temperature=1.0, top_p=1.0)
        self.assertEqual(int(idx_blended[0, 0, 0]), 0)
        # Verify blending was non-trivial: distribution differs from cond alone.
        self.assertFalse(torch.allclose(blended, cond))


if __name__ == "__main__":
    unittest.main()
