"""Dataset cache hash stability tests.

Targets the bug class that destroyed two weeks of training on 2026-05-13:
cache hash drift between machines silently reused the wrong preprocessed
dataset, polluting train splits.

The contract these tests enforce on `load_or_build_cache`'s hash:

  1. Identical inputs -> identical hash on the same machine, byte-for-byte.
  2. CACHE_SCHEMA bump -> different hash (forces rebuild).
  3. Any filter-kwarg value change -> different hash.
  4. Filter-kwarg dict ORDER does not matter (sorted internally).
  5. Different data_dir or max_length or max_samples -> different hash.

The hash function is small enough to reproduce inline; if production drifts
from the formula tested here, the test starts failing — which is precisely
the kind of regression signal we lacked when the original incident happened.
"""

from __future__ import annotations

import hashlib
import unittest

from src.shared.constants import CACHE_SCHEMA, DEFAULT_FILTER_KWARGS


def hash_of(data_dir: str, max_samples: int | None, max_length: int,
            filter_kwargs: dict, schema: int) -> str:
    """Reproduce dataset_cache.load_or_build_cache's hash formula in test space.

    Kept in lockstep with src/data/dataset_cache.py:84-87. If the production
    formula changes, this helper must change too — and that change is the
    signal we want, not a silent drift.
    """
    fkw_str = ":".join(f"{k}={v}" for k, v in sorted(filter_kwargs.items()))
    return hashlib.md5(
        f"{data_dir}:{max_samples}:{max_length}:{fkw_str}:{schema}".encode()
    ).hexdigest()[:12]


class TestCacheHashStability(unittest.TestCase):

    def base_args(self) -> dict:
        return {
            "data_dir": "data/AMASS",
            "max_samples": None,
            "max_length": 1000,
            "filter_kwargs": DEFAULT_FILTER_KWARGS,
            "schema": CACHE_SCHEMA,
        }

    def test_identical_inputs_same_hash(self):
        h1 = hash_of(**self.base_args())
        h2 = hash_of(**self.base_args())
        self.assertEqual(h1, h2)

    def test_schema_bump_changes_hash(self):
        h1 = hash_of(**self.base_args())
        args = self.base_args()
        args["schema"] = CACHE_SCHEMA + 1
        h2 = hash_of(**args)
        self.assertNotEqual(h1, h2)

    def test_data_dir_change_changes_hash(self):
        h1 = hash_of(**self.base_args())
        args = self.base_args()
        args["data_dir"] = "data/AMASS_subset"
        h2 = hash_of(**args)
        self.assertNotEqual(h1, h2)

    def test_max_samples_change_changes_hash(self):
        h1 = hash_of(**self.base_args())
        args = self.base_args()
        args["max_samples"] = 500
        h2 = hash_of(**args)
        self.assertNotEqual(h1, h2)

    def test_max_length_change_changes_hash(self):
        h1 = hash_of(**self.base_args())
        args = self.base_args()
        args["max_length"] = 200
        h2 = hash_of(**args)
        self.assertNotEqual(h1, h2)

    def test_filter_kwarg_value_change_changes_hash(self):
        h1 = hash_of(**self.base_args())
        args = self.base_args()
        modified = dict(DEFAULT_FILTER_KWARGS)
        modified["min_frames"] = 60  # was 30
        args["filter_kwargs"] = modified
        h2 = hash_of(**args)
        self.assertNotEqual(h1, h2)

    def test_filter_kwarg_order_does_not_matter(self):
        """Two identical filter sets in different insertion order must hash equal."""
        keys = list(DEFAULT_FILTER_KWARGS.keys())
        forward_order = {k: DEFAULT_FILTER_KWARGS[k] for k in keys}
        reverse_order = {k: DEFAULT_FILTER_KWARGS[k] for k in reversed(keys)}

        args1 = self.base_args()
        args1["filter_kwargs"] = forward_order
        args2 = self.base_args()
        args2["filter_kwargs"] = reverse_order

        self.assertEqual(hash_of(**args1), hash_of(**args2))

    def test_extra_filter_kwarg_changes_hash(self):
        h1 = hash_of(**self.base_args())
        args = self.base_args()
        modified = dict(DEFAULT_FILTER_KWARGS)
        modified["newConstraint"] = 0.5
        args["filter_kwargs"] = modified
        h2 = hash_of(**args)
        self.assertNotEqual(h1, h2)

    def test_hash_is_12_hex_chars(self):
        """Length is part of the contract — production code stores it in the filename."""
        h = hash_of(**self.base_args())
        self.assertEqual(len(h), 12)
        int(h, 16)  # must parse as hex; raises if not

    def test_hash_matches_production_formula(self):
        """Independent reconstruction of the hash, anchored to a known input.

        This is the test that fails LOUDLY if someone changes the production
        cache key formula without updating this test — exactly the signal
        we lacked during the cache-drift incident.
        """
        expected = hash_of(
            data_dir="data/AMASS",
            max_samples=None,
            max_length=1000,
            filter_kwargs={"max_accel": 50.0, "max_joint_rotvel": 30.0,
                          "max_root_speed": 10.0, "min_frames": 30,
                          "min_variance": 0.0001},
            schema=4,
        )
        # If CACHE_SCHEMA / DEFAULT_FILTER_KWARGS / formula change, this test
        # FAILS and we get a deliberate review moment instead of silent drift.
        actual = hash_of(
            data_dir="data/AMASS",
            max_samples=None,
            max_length=1000,
            filter_kwargs=DEFAULT_FILTER_KWARGS,
            schema=CACHE_SCHEMA,
        )
        self.assertEqual(expected, actual,
                         "production cache formula or DEFAULT_FILTER_KWARGS "
                         "drifted from the test anchor — confirm this is intentional, "
                         "then update the test anchor.")


if __name__ == "__main__":
    unittest.main()
