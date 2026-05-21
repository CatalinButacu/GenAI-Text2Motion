"""Contract tests for ``seed_all`` — the single-entry reproducibility helper.

Any drift here breaks ablations and makes published numbers unreproducible.
"""

from __future__ import annotations

import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.modules.motion.training.trainer_utils import lock_seed
from src.shared.seed import seed_all, seed_dict, seed_worker


class TestSeedAll(unittest.TestCase):

    def test_python_random_seeded(self):
        seed_all(42)
        a = [random.random() for _ in range(5)]
        seed_all(42)
        b = [random.random() for _ in range(5)]
        self.assertEqual(a, b)

    def test_numpy_seeded(self):
        seed_all(42)
        a = np.random.standard_normal(8)
        seed_all(42)
        b = np.random.standard_normal(8)
        np.testing.assert_array_equal(a, b)

    def test_torch_seeded(self):
        seed_all(42)
        a = torch.randn(8)
        seed_all(42)
        b = torch.randn(8)
        self.assertTrue(torch.equal(a, b))

    def test_different_seeds_different_streams(self):
        seed_all(1)
        a = torch.randn(8)
        seed_all(2)
        b = torch.randn(8)
        self.assertFalse(torch.equal(a, b))

    def test_deterministic_mode_sets_env_vars(self):
        env_backup = dict(os.environ)
        try:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            seed_all(42, deterministic=True)
            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
            # Restore default torch deterministic state so other tests aren't surprised
            torch.use_deterministic_algorithms(False, warn_only=True)

    def test_non_deterministic_default_does_not_set_cublas(self):
        env_backup = dict(os.environ)
        try:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            seed_all(42, deterministic=False)
            self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


class TestSeedWorker(unittest.TestCase):

    def test_different_workers_get_different_seeds(self):
        with patch("torch.initial_seed", return_value=12345):
            seed_worker(0)
            v0 = np.random.standard_normal(1)[0]
            seed_worker(1)
            v1 = np.random.standard_normal(1)[0]
            self.assertNotEqual(v0, v1)

    def test_same_worker_id_reproducible(self):
        with patch("torch.initial_seed", return_value=12345):
            seed_worker(0)
            a = np.random.standard_normal(4)
            seed_worker(0)
            b = np.random.standard_normal(4)
            np.testing.assert_array_equal(a, b)


class TestSeedDict(unittest.TestCase):

    def test_contains_required_keys(self):
        d = seed_dict(42, deterministic=False)
        for key in (
            "seed", "deterministic", "torch_version", "cuda_available",
            "cudnn_deterministic", "cudnn_benchmark",
            "pythonhashseed", "cublas_workspace_config",
        ):
            self.assertIn(key, d)

    def test_seed_value_preserved(self):
        d = seed_dict(99, deterministic=True)
        self.assertEqual(d["seed"], 99)
        self.assertTrue(d["deterministic"])


class TestLockSeedBackwardCompat(unittest.TestCase):
    """The old lock_seed name still works for callers that haven't migrated."""

    def test_lock_seed_still_seeds_torch(self):
        lock_seed(42)
        a = torch.randn(4)
        lock_seed(42)
        b = torch.randn(4)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
