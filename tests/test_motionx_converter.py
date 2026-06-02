"""Unit tests for the Motion-X smplx-322 -> repo 168 mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data.motionx_to_smplx168 import (  # noqa: E402
    MOTIONX_DIM,
    REPO_DIM,
    map_motionx_322_to_168,
)


class TestMotionXSliceMap(unittest.TestCase):

    def test_shape_is_repo_dim(self):
        arr = np.zeros((10, MOTIONX_DIM), dtype=np.float32)
        out = map_motionx_322_to_168(arr)
        self.assertEqual(out.shape, (10, REPO_DIM))

    def test_root_orient_preserved(self):
        arr = np.zeros((3, MOTIONX_DIM), dtype=np.float32)
        arr[:, 0:3] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        out = map_motionx_322_to_168(arr)
        np.testing.assert_allclose(out[:, 0:3], arr[:, 0:3])

    def test_transl_comes_from_offset_215(self):
        arr = np.zeros((2, MOTIONX_DIM), dtype=np.float32)
        arr[:, 215:218] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = map_motionx_322_to_168(arr)
        np.testing.assert_allclose(out[:, 3:6], arr[:, 215:218])

    def test_body_pose_preserved(self):
        arr = np.zeros((2, MOTIONX_DIM), dtype=np.float32)
        arr[:, 3:66] = np.arange(126, dtype=np.float32).reshape(2, 63)
        out = map_motionx_322_to_168(arr)
        np.testing.assert_allclose(out[:, 6:69], arr[:, 3:66])

    def test_left_hand_preserved(self):
        arr = np.zeros((1, MOTIONX_DIM), dtype=np.float32)
        arr[0, 66:111] = np.arange(45, dtype=np.float32)
        out = map_motionx_322_to_168(arr)
        np.testing.assert_allclose(out[0, 69:114], np.arange(45, dtype=np.float32))

    def test_right_hand_preserved(self):
        arr = np.zeros((1, MOTIONX_DIM), dtype=np.float32)
        arr[0, 111:156] = np.arange(45, dtype=np.float32) + 100
        out = map_motionx_322_to_168(arr)
        np.testing.assert_allclose(
            out[0, 114:159], np.arange(45, dtype=np.float32) + 100,
        )

    def test_jaw_eyes_preserved(self):
        arr = np.zeros((1, MOTIONX_DIM), dtype=np.float32)
        arr[0, 156:165] = np.arange(9, dtype=np.float32)
        out = map_motionx_322_to_168(arr)
        np.testing.assert_allclose(out[0, 159:168], np.arange(9, dtype=np.float32))

    def test_wrong_dim_raises(self):
        with self.assertRaises(ValueError):
            map_motionx_322_to_168(np.zeros((10, 100), dtype=np.float32))

    def test_face_params_not_leaked(self):
        """Expression / face_shape / betas must NOT appear anywhere in 168 output."""
        arr = np.zeros((2, MOTIONX_DIM), dtype=np.float32)
        arr[:, 165:215] = 7.7   # expression
        arr[:, 218:228] = 3.3   # betas
        arr[:, 228:322] = 9.9   # face_shape
        out = map_motionx_322_to_168(arr)
        self.assertFalse(np.any(out == 7.7))
        self.assertFalse(np.any(out == 3.3))
        self.assertFalse(np.any(out == 9.9))


if __name__ == "__main__":
    unittest.main()
