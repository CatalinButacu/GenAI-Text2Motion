"""Determinism contract for the SSM forward + understanding pipeline.

A regression where someone introduces a non-deterministic op (CUDA-style atomic
add, unseeded dropout, hash-randomized dict iteration, etc.) silently breaks
reproducibility — which makes ablation studies and checkpoint comparisons
worthless. These tests guard the core invariant: with the same seed and same
input, we get the same output.

Tests run on CPU with tiny configs; no GPU, no checkpoints, no data files.
"""

from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch

from src.modules.motion.ssm import BiMambaLayer, MambaLayer, SSMConfig
from src.modules.understanding.spacy import SpacyParser


def tensorHash(t: torch.Tensor) -> str:
    """Stable hash of a tensor's float32 bytes — useful for diffing across runs."""
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


def lockTorchSeed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CPU only; full determinism not needed


class TestSsmForwardDeterminism(unittest.TestCase):

    def configFor(self, dModel: int = 16, dState: int = 4) -> SSMConfig:
        return SSMConfig(dModel=dModel, dState=dState, dConv=4, expand=2)

    def test_same_seed_same_output(self):
        cfg = self.configFor()
        lockTorchSeed(123)
        layer1 = MambaLayer(cfg)
        x = torch.randn(2, 8, cfg.dModel)
        out1 = layer1(x)

        lockTorchSeed(123)
        layer2 = MambaLayer(cfg)
        out2 = layer2(x)

        self.assertEqual(tensorHash(out1), tensorHash(out2))

    def test_different_seed_different_output(self):
        cfg = self.configFor()
        lockTorchSeed(1)
        layer1 = MambaLayer(cfg)
        lockTorchSeed(2)
        layer2 = MambaLayer(cfg)
        x = torch.randn(2, 8, cfg.dModel)
        out1 = layer1(x)
        out2 = layer2(x)

        self.assertNotEqual(tensorHash(out1), tensorHash(out2))

    def test_eval_mode_is_deterministic_across_calls(self):
        cfg = self.configFor()
        lockTorchSeed(99)
        layer = MambaLayer(cfg).eval()
        x = torch.randn(2, 8, cfg.dModel)

        with torch.no_grad():
            outA = layer(x)
            outB = layer(x)

        self.assertTrue(torch.equal(outA, outB))

    def test_bimamba_same_seed_same_output(self):
        cfg = self.configFor()
        lockTorchSeed(7)
        layer1 = BiMambaLayer(cfg)
        x = torch.randn(2, 6, cfg.dModel)
        out1 = layer1(x)

        lockTorchSeed(7)
        layer2 = BiMambaLayer(cfg)
        out2 = layer2(x)

        self.assertEqual(tensorHash(out1), tensorHash(out2))


class TestSsmScanNumericalStability(unittest.TestCase):
    """Forward pass must not produce NaN/inf on extreme inputs."""

    def test_zero_input_yields_finite_output(self):
        cfg = SSMConfig(dModel=16, dState=4, dConv=4, expand=2)
        layer = MambaLayer(cfg).eval()
        x = torch.zeros(2, 8, cfg.dModel)

        with torch.no_grad():
            out = layer(x)

        self.assertTrue(torch.isfinite(out).all().item())

    def test_large_input_does_not_explode(self):
        cfg = SSMConfig(dModel=16, dState=4, dConv=4, expand=2)
        layer = MambaLayer(cfg).eval()
        x = torch.randn(2, 8, cfg.dModel) * 50.0

        with torch.no_grad():
            out = layer(x)

        self.assertTrue(torch.isfinite(out).all().item(),
                        "large input produced NaN or inf in SSM forward")

    def test_long_sequence_state_stays_bounded(self):
        cfg = SSMConfig(dModel=8, dState=4, dConv=4, expand=2)
        layer = MambaLayer(cfg).eval()
        x = torch.randn(1, 200, cfg.dModel) * 0.5

        with torch.no_grad():
            out = layer(x)

        self.assertTrue(torch.isfinite(out).all().item())
        maxAbs = float(out.abs().max())
        self.assertLess(maxAbs, 1e3,
                        f"long-sequence output magnitude unbounded: {maxAbs}")


class TestUnderstandingDeterminism(unittest.TestCase):
    """M1 (spaCy parser) is rule-based and must be byte-identical across calls."""

    def test_parse_same_prompt_same_result(self):
        parser = SpacyParser()
        prompt = "a person walks forward then sits down"
        a = parser.parse(prompt)
        b = parser.parse(prompt)

        self.assertEqual(len(a.entities), len(b.entities))
        self.assertEqual(len(a.actions), len(b.actions))
        for ea, eb in zip(a.entities, b.entities, strict=True):
            self.assertEqual(ea.objectType, eb.objectType)
            self.assertEqual(ea.name, eb.name)
        for aa, ab in zip(a.actions, b.actions, strict=True):
            self.assertEqual(aa.actionType, ab.actionType)
            self.assertEqual(aa.order, ab.order)


class TestNumpySeedingContract(unittest.TestCase):
    """Augmentation generators must be deterministic under explicit seed."""

    def test_same_numpy_seed_same_sequence(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        seq1 = rng1.standard_normal(100)
        seq2 = rng2.standard_normal(100)
        np.testing.assert_array_equal(seq1, seq2)


if __name__ == "__main__":
    unittest.main()
