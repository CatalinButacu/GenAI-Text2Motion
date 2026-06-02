"""Motion module tests.

Uses random tensors and mocked checkpoints. No GPU or trained weights needed.
SSM architecture tests verify shapes only (forward pass with random weights).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.architecture.ssm import (
    BiMambaLayer,
    MambaLayer,
    SSMConfig,
    create_ssm_layer,
    get_ssm_info,
)
from src.modules.motion.models import MotionClip, MotionSource


class TestSSMConfig(unittest.TestCase):
    def test_d_inner_computed(self):
        cfg = SSMConfig(d_model=64, expand=2)
        self.assertEqual(cfg.d_inner, 128)

    def test_dt_rank_auto(self):
        cfg = SSMConfig(d_model=64)
        assert isinstance(cfg.dt_rank, int)
        self.assertGreater(cfg.dt_rank, 0)

    def test_dt_rank_explicit(self):
        cfg = SSMConfig(d_model=64, dt_rank=8)
        self.assertEqual(cfg.dt_rank, 8)


class TestMambaLayer(unittest.TestCase):
    def _cfg(self, d_model=32, d_state=4):
        return SSMConfig(d_model=d_model, d_state=d_state, d_conv=4, expand=2)

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
        h = torch.zeros(2, cfg.d_inner, cfg.d_state)
        out, h_new, _ = layer.step(x_t, h)
        self.assertEqual(out.shape, (2, 32))
        self.assertEqual(h_new.shape, (2, cfg.d_inner, cfg.d_state))


class TestBiMambaLayer(unittest.TestCase):
    def test_forward_shape(self):
        cfg = SSMConfig(d_model=32, d_state=4, d_conv=4, expand=2)
        layer = BiMambaLayer(cfg)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 32))


class TestFactory(unittest.TestCase):
    def test_create_mamba_layer(self):
        layer = create_ssm_layer("mamba", d_model=32, d_state=4)
        self.assertIsInstance(layer, MambaLayer)

    def test_unknown_layer_raises(self):
        with self.assertRaises(ValueError):
            create_ssm_layer("unknown_type")

    def test_get_ssm_info_keys(self):
        info = get_ssm_info()
        self.assertIn("layers", info)
        self.assertIn("mamba", info["layers"])


class TestMotionClip(unittest.TestCase):
    def test_duration(self):
        clip = MotionClip(action="walk", smplx_params=np.zeros((60, 168)), fps=30)
        self.assertAlmostEqual(clip.duration, 2.0)

    def test_num_frames(self):
        clip = MotionClip(action="walk", smplx_params=np.zeros((45, 168)))
        self.assertEqual(clip.num_frames, 45)

    def test_source_default(self):
        clip = MotionClip(action="run", smplx_params=np.zeros((10, 168)))
        self.assertEqual(clip.source, MotionSource.SSM)


class TestMotionGenerator(unittest.TestCase):
    @patch("src.modules.motion.generator.SSMMotionModel")
    def test_generate_returns_clip(self, MockModel):
        from src.modules.motion.generator import MotionGenerator
        from src.shared.config import MotionConfig

        dummy_clip = MotionClip(action="walk", smplx_params=np.zeros((30, 168)))
        instance = MockModel.return_value
        instance.generate_from_text_tokens.return_value = dummy_clip

        gen = MotionGenerator.__new__(MotionGenerator)
        gen.backend = instance
        gen.cfg = MotionConfig()
        gen.reranker = None
        gen.retriever = None

        result = gen.generate("a person walks", num_frames=30)
        self.assertIsInstance(result, MotionClip)
        instance.generate_from_text_tokens.assert_called_once_with(
            "a person walks", 30, temperature=1.0, top_p=1.0, cfg_scale=2.0
        )


if __name__ == "__main__":
    unittest.main()
