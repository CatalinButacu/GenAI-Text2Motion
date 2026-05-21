"""Motion module tests.

Uses random tensors and mocked checkpoints. No GPU or trained weights needed.
SSM architecture tests verify shapes only (forward pass with random weights).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.modules.motion.models import MotionClip, MotionSource
from src.modules.motion.ssm import (
    BiMambaLayer,
    MambaLayer,
    SSMConfig,
    createSsmLayer,
    getSsmInfo,
)


class TestSSMConfig(unittest.TestCase):
    def test_d_inner_computed(self):
        cfg = SSMConfig(dModel=64, expand=2)
        self.assertEqual(cfg.d_inner, 128)

    def test_dt_rank_auto(self):
        cfg = SSMConfig(dModel=64)
        assert isinstance(cfg.dtRank, int)
        self.assertGreater(cfg.dtRank, 0)

    def test_dt_rank_explicit(self):
        cfg = SSMConfig(dModel=64, dtRank=8)
        self.assertEqual(cfg.dtRank, 8)


class TestMambaLayer(unittest.TestCase):
    def _cfg(self, dModel=32, dState=4):
        return SSMConfig(dModel=dModel, dState=dState, dConv=4, expand=2)

    def test_forward_shape(self):
        cfg = self._cfg()
        layer = MambaLayer(cfg)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_step_matches_forward_shape(self):
        cfg = self._cfg()
        layer = MambaLayer(cfg)
        layer.eval()
        x_t = torch.randn(2, 32)
        h = torch.zeros(2, cfg.d_inner, cfg.dState)
        out, h_new, buf = layer.step(x_t, h)
        self.assertEqual(out.shape, (2, 32))
        self.assertEqual(h_new.shape, (2, cfg.d_inner, cfg.dState))


class TestBiMambaLayer(unittest.TestCase):
    def test_forward_shape(self):
        cfg = SSMConfig(dModel=32, dState=4, dConv=4, expand=2)
        layer = BiMambaLayer(cfg)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 32))


class TestFactory(unittest.TestCase):
    def test_create_mamba_layer(self):
        layer = createSsmLayer("mamba", dModel=32, dState=4)
        self.assertIsInstance(layer, MambaLayer)

    def test_unknown_layer_raises(self):
        with self.assertRaises(ValueError):
            createSsmLayer("unknown_type")

    def test_get_ssm_info_keys(self):
        info = getSsmInfo()
        self.assertIn("layers", info)
        self.assertIn("mamba", info["layers"])


class TestMotionClip(unittest.TestCase):
    def test_duration(self):
        clip = MotionClip(action="walk", smplxParams=np.zeros((60, 168)), fps=30)
        self.assertAlmostEqual(clip.duration, 2.0)

    def test_num_frames(self):
        clip = MotionClip(action="walk", smplxParams=np.zeros((45, 168)))
        self.assertEqual(clip.numFrames, 45)

    def test_source_default(self):
        clip = MotionClip(action="run", smplxParams=np.zeros((10, 168)))
        self.assertEqual(clip.source, MotionSource.SSM)


class TestMotionGenerator(unittest.TestCase):
    @patch("src.modules.motion.generator.SSMMotionModel")
    def test_generate_returns_clip(self, MockModel):
        dummy_clip = MotionClip(action="walk", smplxParams=np.zeros((30, 168)))
        instance = MockModel.return_value
        instance.generateFromTextTokens.return_value = dummy_clip

        from src.modules.motion.generator import MotionGenerator

        gen = MotionGenerator.__new__(MotionGenerator)
        gen.backend = instance
        gen.temperature = 1.0
        gen.topP = 1.0
        result = gen.generate("a person walks", numFrames=30)
        self.assertIsInstance(result, MotionClip)
        instance.generateFromTextTokens.assert_called_once_with(
            "a person walks", 30, temperature=1.0, topP=1.0
        )


if __name__ == "__main__":
    unittest.main()
