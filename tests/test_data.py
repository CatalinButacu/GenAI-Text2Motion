"""Data layer tests: normalize/denormalize round-trip + augmentation invariants.

Pure numpy; no torch, no checkpoints, no GPU. Fills the coverage hole flagged
by the 2026-05 audit (src/data/* had zero direct tests).
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

from src.data.augmentation import (
    LR_SWAP,
    AugmentationPipeline,
    add_noise,
    mirror_flip,
    mirror_flip_text,
    resample_to_fps,
    speed_perturbation,
    temporal_crop,
)
from src.data.motion_normalize import (
    MotionStats,
    compute_motion_stats,
    denormalize,
    normalize,
)


def make_stats(motion_dim: int = 168) -> MotionStats:
    rng = np.random.default_rng(0)
    return MotionStats(
        mean=rng.standard_normal(motion_dim).astype(np.float32),
        std=(rng.uniform(0.5, 1.5, motion_dim)).astype(np.float32),
    )


class TestMotionNormalize(unittest.TestCase):

    def test_round_trip_recovers_input(self):
        stats = make_stats()
        rng = np.random.default_rng(1)
        motion = rng.standard_normal((50, 168)).astype(np.float32) * 0.1
        z = normalize(motion, stats, clip_value=None)
        back = denormalize(z, stats)
        np.testing.assert_allclose(back, motion, atol=1e-5)

    def test_clip_value_bounds_output(self):
        stats = make_stats()
        rng = np.random.default_rng(2)
        motion = rng.standard_normal((50, 168)).astype(np.float32) * 100.0
        z = normalize(motion, stats, clip_value=5.0)
        self.assertLessEqual(float(z.max()), 5.0 + 1e-6)
        self.assertGreaterEqual(float(z.min()), -5.0 - 1e-6)

    def test_zero_variance_channel_does_not_explode(self):
        samples = [{"motion": np.ones((4, 10), dtype=np.float32)}]
        computed = compute_motion_stats(samples)
        self.assertTrue(np.all(computed.std > 0.0))

    def test_translation_stats_override(self):
        stats = make_stats()
        trans_stats = MotionStats(
            mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            std=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        motion = np.zeros((10, 168), dtype=np.float32)
        motion[:, 3:6] = np.array([[1.0, 2.0, 3.0]] * 10, dtype=np.float32)
        z = normalize(motion, stats, clip_value=None, trans_stats=trans_stats)
        np.testing.assert_allclose(z[:, 3:6], 0.0, atol=1e-6)

    def test_compute_stats_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_motion_stats([])


class TestMirrorFlip(unittest.TestCase):

    def test_text_swap_round_trip(self):
        original = "raise the left arm and step right"
        flipped = mirror_flip_text(original)
        self.assertNotEqual(flipped, original)
        self.assertEqual(mirror_flip_text(flipped), original)

    def test_text_swap_table_complete(self):
        for src, dst in LR_SWAP.items():
            self.assertEqual(LR_SWAP[dst], src)

    def test_motion_short_input_passthrough(self):
        m = np.zeros((10, 50), dtype=np.float32)
        out = mirror_flip(m)
        np.testing.assert_array_equal(out, m)

    def test_motion_empty_passthrough(self):
        m = np.zeros((0, 168), dtype=np.float32)
        out = mirror_flip(m)
        self.assertEqual(out.shape[0], 0)

    def test_motion_translation_x_negated(self):
        m = np.zeros((5, 168), dtype=np.float32)
        m[:, 3] = 2.5  # tx
        out = mirror_flip(m)
        np.testing.assert_allclose(out[:, 3], -2.5)

    def test_motion_double_flip_is_identity(self):
        rng = np.random.default_rng(3)
        m = rng.standard_normal((20, 168)).astype(np.float32) * 0.1
        once = mirror_flip(m)
        twice = mirror_flip(once)
        np.testing.assert_allclose(twice, m, atol=1e-5)


class TestTemporalOps(unittest.TestCase):

    def test_crop_to_length_when_longer(self):
        m = np.arange(500 * 168).reshape(500, 168).astype(np.float32)
        rng = np.random.default_rng(4)
        cropped = temporal_crop(m, max_length=120, rng=rng)
        self.assertEqual(cropped.shape, (120, 168))

    def test_crop_passes_short_motion_through(self):
        m = np.zeros((80, 168), dtype=np.float32)
        cropped = temporal_crop(m, max_length=200)
        self.assertEqual(cropped.shape, (80, 168))

    def test_speed_perturb_changes_length(self):
        rng = np.random.default_rng(5)
        m = np.zeros((100, 168), dtype=np.float32)
        out = speed_perturbation(m, rng, speed_range=(0.5, 0.5))  # exactly 2x slower
        self.assertEqual(out.shape[0], 200)

    def test_speed_perturb_near_unity_is_passthrough(self):
        rng = np.random.default_rng(6)
        m = np.zeros((100, 168), dtype=np.float32)
        out = speed_perturbation(m, rng, speed_range=(1.0, 1.0))
        self.assertEqual(out.shape, m.shape)

    def test_add_noise_preserves_translation(self):
        rng = np.random.default_rng(7)
        m = np.ones((20, 168), dtype=np.float32)
        out = add_noise(m, sigma=0.1, rng=rng)
        np.testing.assert_array_equal(out[:, 3:6], m[:, 3:6])
        diff = np.abs(out[:, 6:] - m[:, 6:])
        self.assertGreater(float(diff.mean()), 0.0)


class TestAugmentationPipeline(unittest.TestCase):

    def test_pipeline_returns_array(self):
        pipeline = AugmentationPipeline(seed=42, max_length=64)
        m = np.zeros((100, 168), dtype=np.float32)
        out = pipeline(m)
        self.assertIsInstance(out, np.ndarray)
        self.assertLessEqual(out.shape[0], 200)

    def test_pipeline_deterministic_with_seed(self):
        m = np.ones((100, 168), dtype=np.float32)
        out1 = AugmentationPipeline(seed=123, max_length=80)(m.copy())
        out2 = AugmentationPipeline(seed=123, max_length=80)(m.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_pipeline_invoke_matches_call(self):
        pipeline = AugmentationPipeline(seed=99)
        m = np.zeros((50, 168), dtype=np.float32)
        a = pipeline(m.copy())
        b = AugmentationPipeline(seed=99).invoke(m.copy())
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("frames", [1, 4, 30, 100, 300])
@pytest.mark.parametrize("motion_dim", [6, 168])
def test_property_mirrorflip_shape_preserved(frames: int, motion_dim: int) -> None:
    """mirror_flip must never alter the input shape regardless of size."""
    rng = np.random.default_rng(frames * motion_dim)
    m = rng.standard_normal((frames, motion_dim)).astype(np.float32) * 0.1
    out = mirror_flip(m)
    assert out.shape == m.shape


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_property_mirrorflip_full_smplx_is_involutive(seed: int) -> None:
    """For full 168-d SMPL-X poses, applying mirror_flip twice recovers the input."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((20, 168)).astype(np.float32) * 0.1
    twice = mirror_flip(mirror_flip(m))
    np.testing.assert_allclose(twice, m, atol=1e-5)


@pytest.mark.parametrize("frames,src_fps,tgt_fps", [
    (60, 30.0, 60.0),
    (60, 60.0, 30.0),
    (100, 120.0, 30.0),
    (45, 24.0, 30.0),
])
def test_property_resample_changes_duration_consistently(frames: int, src_fps: float,
                                                          tgt_fps: float) -> None:
    """Resampling preserves wall-clock duration to within one frame."""
    m = np.zeros((frames, 168), dtype=np.float32)
    out = resample_to_fps(m, src_fps=src_fps, tgt_fps=tgt_fps)
    expected = round((frames / src_fps) * tgt_fps)
    assert abs(out.shape[0] - expected) <= 1, (
        f"resample {src_fps}->{tgt_fps}: got {out.shape[0]} frames, "
        f"expected ~{expected}"
    )


@pytest.mark.parametrize("bad_fps", [0.0, 0.5, 1.0, -10.0, float("nan"), float("inf")])
def test_property_resample_rejects_degenerate_fps(bad_fps: float) -> None:
    """Degenerate src fps must return input unchanged — never crash, never NaN out."""
    m = np.ones((50, 168), dtype=np.float32)
    out = resample_to_fps(m, src_fps=bad_fps, tgt_fps=30.0)
    assert out.shape == m.shape
    assert np.isfinite(out).all()


@pytest.mark.parametrize("sigma", [0.0, 1e-4, 0.01, 0.1, 1.0])
def test_property_noise_preserves_shape_and_finiteness(sigma: float) -> None:
    """add_noise must never alter shape or introduce NaN, across sigma magnitudes."""
    rng = np.random.default_rng(0)
    m = np.ones((30, 168), dtype=np.float32)
    out = add_noise(m, sigma=sigma, rng=rng)
    assert out.shape == m.shape
    assert np.isfinite(out).all()


@pytest.mark.parametrize("speed_factor,input_frames", [
    (0.5, 100),   # 2x slower
    (1.0, 100),   # no change
    (1.5, 100),
    (2.0, 100),   # 2x faster
    (0.5, 1),     # degenerate input — must not crash
    (2.0, 2),
])
def test_property_speed_perturb_never_crashes_or_nans(speed_factor: float,
                                                       input_frames: int) -> None:
    rng = np.random.default_rng(0)
    m = np.zeros((input_frames, 168), dtype=np.float32)
    out = speed_perturbation(m, rng, speed_range=(speed_factor, speed_factor))
    assert np.isfinite(out).all()
    assert out.shape[1] == 168


if __name__ == "__main__":
    unittest.main()
