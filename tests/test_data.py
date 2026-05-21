"""Data layer tests: normalize/denormalize round-trip + augmentation invariants.

Pure numpy; no torch, no checkpoints, no GPU. Fills the coverage hole flagged
by the 2026-05 audit (src/data/* had zero direct tests).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.data.augmentation import (
    LR_SWAP,
    AugmentationPipeline,
    addNoise,
    mirrorFlip,
    mirrorFlipText,
    speedPerturbation,
    temporalCrop,
)
from src.data.motion_normalize import (
    MotionStats,
    computeMotionStats,
    denormalize,
    normalize,
)


def makeStats(motionDim: int = 168) -> MotionStats:
    rng = np.random.default_rng(0)
    return MotionStats(
        mean=rng.standard_normal(motionDim).astype(np.float32),
        std=(rng.uniform(0.5, 1.5, motionDim)).astype(np.float32),
    )


class TestMotionNormalize(unittest.TestCase):

    def test_round_trip_recovers_input(self):
        stats = makeStats()
        rng = np.random.default_rng(1)
        motion = rng.standard_normal((50, 168)).astype(np.float32) * 0.1
        z = normalize(motion, stats, clipValue=None)
        back = denormalize(z, stats)
        np.testing.assert_allclose(back, motion, atol=1e-5)

    def test_clip_value_bounds_output(self):
        stats = makeStats()
        rng = np.random.default_rng(2)
        motion = rng.standard_normal((50, 168)).astype(np.float32) * 100.0
        z = normalize(motion, stats, clipValue=5.0)
        self.assertLessEqual(float(z.max()), 5.0 + 1e-6)
        self.assertGreaterEqual(float(z.min()), -5.0 - 1e-6)

    def test_zero_variance_channel_does_not_explode(self):
        samples = [{"motion": np.ones((4, 10), dtype=np.float32)}]
        computed = computeMotionStats(samples)
        self.assertTrue(np.all(computed.std > 0.0))

    def test_translation_stats_override(self):
        stats = makeStats()
        transStats = MotionStats(
            mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            std=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        motion = np.zeros((10, 168), dtype=np.float32)
        motion[:, 3:6] = np.array([[1.0, 2.0, 3.0]] * 10, dtype=np.float32)
        z = normalize(motion, stats, clipValue=None, transStats=transStats)
        np.testing.assert_allclose(z[:, 3:6], 0.0, atol=1e-6)

    def test_compute_stats_empty_raises(self):
        with self.assertRaises(ValueError):
            computeMotionStats([])


class TestMirrorFlip(unittest.TestCase):

    def test_text_swap_round_trip(self):
        original = "raise the left arm and step right"
        flipped = mirrorFlipText(original)
        self.assertNotEqual(flipped, original)
        self.assertEqual(mirrorFlipText(flipped), original)

    def test_text_swap_table_complete(self):
        for src, dst in LR_SWAP.items():
            self.assertEqual(LR_SWAP[dst], src)

    def test_motion_short_input_passthrough(self):
        m = np.zeros((10, 50), dtype=np.float32)
        out = mirrorFlip(m)
        np.testing.assert_array_equal(out, m)

    def test_motion_empty_passthrough(self):
        m = np.zeros((0, 168), dtype=np.float32)
        out = mirrorFlip(m)
        self.assertEqual(out.shape[0], 0)

    def test_motion_translation_x_negated(self):
        m = np.zeros((5, 168), dtype=np.float32)
        m[:, 3] = 2.5  # tx
        out = mirrorFlip(m)
        np.testing.assert_allclose(out[:, 3], -2.5)

    def test_motion_double_flip_is_identity(self):
        rng = np.random.default_rng(3)
        m = rng.standard_normal((20, 168)).astype(np.float32) * 0.1
        once = mirrorFlip(m)
        twice = mirrorFlip(once)
        np.testing.assert_allclose(twice, m, atol=1e-5)


class TestTemporalOps(unittest.TestCase):

    def test_crop_to_length_when_longer(self):
        m = np.arange(500 * 168).reshape(500, 168).astype(np.float32)
        rng = np.random.default_rng(4)
        cropped = temporalCrop(m, maxLength=120, rng=rng)
        self.assertEqual(cropped.shape, (120, 168))

    def test_crop_passes_short_motion_through(self):
        m = np.zeros((80, 168), dtype=np.float32)
        cropped = temporalCrop(m, maxLength=200)
        self.assertEqual(cropped.shape, (80, 168))

    def test_speed_perturb_changes_length(self):
        rng = np.random.default_rng(5)
        m = np.zeros((100, 168), dtype=np.float32)
        out = speedPerturbation(m, rng, speedRange=(0.5, 0.5))  # exactly 2x slower
        self.assertEqual(out.shape[0], 200)

    def test_speed_perturb_near_unity_is_passthrough(self):
        rng = np.random.default_rng(6)
        m = np.zeros((100, 168), dtype=np.float32)
        out = speedPerturbation(m, rng, speedRange=(1.0, 1.0))
        self.assertEqual(out.shape, m.shape)

    def test_add_noise_preserves_translation(self):
        rng = np.random.default_rng(7)
        m = np.ones((20, 168), dtype=np.float32)
        out = addNoise(m, sigma=0.1, rng=rng)
        np.testing.assert_array_equal(out[:, 3:6], m[:, 3:6])
        diff = np.abs(out[:, 6:] - m[:, 6:])
        self.assertGreater(float(diff.mean()), 0.0)


class TestAugmentationPipeline(unittest.TestCase):

    def test_pipeline_returns_array(self):
        pipeline = AugmentationPipeline(seed=42, maxLength=64)
        m = np.zeros((100, 168), dtype=np.float32)
        out = pipeline(m)
        self.assertIsInstance(out, np.ndarray)
        self.assertLessEqual(out.shape[0], 200)

    def test_pipeline_deterministic_with_seed(self):
        m = np.ones((100, 168), dtype=np.float32)
        out1 = AugmentationPipeline(seed=123, maxLength=80)(m.copy())
        out2 = AugmentationPipeline(seed=123, maxLength=80)(m.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_pipeline_invoke_matches_call(self):
        pipeline = AugmentationPipeline(seed=99)
        m = np.zeros((50, 168), dtype=np.float32)
        a = pipeline(m.copy())
        b = AugmentationPipeline(seed=99).invoke(m.copy())
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
