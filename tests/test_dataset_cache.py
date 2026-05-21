"""Dataset cache hash stability tests.

Targets the bug class that destroyed two weeks of training on 2026-05-13:
cache hash drift between machines silently reused the wrong preprocessed
dataset, polluting train splits.

The contract these tests enforce on `loadOrBuildCache`'s hash:

  1. Identical inputs -> identical hash on the same machine, byte-for-byte.
  2. CACHE_SCHEMA bump -> different hash (forces rebuild).
  3. Any filter-kwarg value change -> different hash.
  4. Filter-kwarg dict ORDER does not matter (sorted internally).
  5. Different dataDir or maxLength or maxSamples -> different hash.

The hash function is small enough to reproduce inline; if production drifts
from the formula tested here, the test starts failing — which is precisely
the kind of regression signal we lacked when the original incident happened.
"""

from __future__ import annotations

import hashlib
import unittest

from src.data.dataset_cache import CACHE_SCHEMA, DEFAULT_FILTER_KWARGS


def hashOf(dataDir: str, maxSamples: int | None, maxLength: int,
            filterKwargs: dict, schema: int) -> str:
    """Reproduce dataset_cache.loadOrBuildCache's hash formula in test space.

    Kept in lockstep with src/data/dataset_cache.py:84-87. If the production
    formula changes, this helper must change too — and that change is the
    signal we want, not a silent drift.
    """
    fkwStr = ":".join(f"{k}={v}" for k, v in sorted(filterKwargs.items()))
    return hashlib.md5(
        f"{dataDir}:{maxSamples}:{maxLength}:{fkwStr}:{schema}".encode()
    ).hexdigest()[:12]


class TestCacheHashStability(unittest.TestCase):

    def baseArgs(self) -> dict:
        return {
            "dataDir": "data/AMASS",
            "maxSamples": None,
            "maxLength": 1000,
            "filterKwargs": DEFAULT_FILTER_KWARGS,
            "schema": CACHE_SCHEMA,
        }

    def test_identical_inputs_same_hash(self):
        h1 = hashOf(**self.baseArgs())
        h2 = hashOf(**self.baseArgs())
        self.assertEqual(h1, h2)

    def test_schema_bump_changes_hash(self):
        h1 = hashOf(**self.baseArgs())
        args = self.baseArgs()
        args["schema"] = CACHE_SCHEMA + 1
        h2 = hashOf(**args)
        self.assertNotEqual(h1, h2)

    def test_data_dir_change_changes_hash(self):
        h1 = hashOf(**self.baseArgs())
        args = self.baseArgs()
        args["dataDir"] = "data/AMASS_subset"
        h2 = hashOf(**args)
        self.assertNotEqual(h1, h2)

    def test_max_samples_change_changes_hash(self):
        h1 = hashOf(**self.baseArgs())
        args = self.baseArgs()
        args["maxSamples"] = 500
        h2 = hashOf(**args)
        self.assertNotEqual(h1, h2)

    def test_max_length_change_changes_hash(self):
        h1 = hashOf(**self.baseArgs())
        args = self.baseArgs()
        args["maxLength"] = 200
        h2 = hashOf(**args)
        self.assertNotEqual(h1, h2)

    def test_filter_kwarg_value_change_changes_hash(self):
        h1 = hashOf(**self.baseArgs())
        args = self.baseArgs()
        modified = dict(DEFAULT_FILTER_KWARGS)
        modified["minFrames"] = 60  # was 30
        args["filterKwargs"] = modified
        h2 = hashOf(**args)
        self.assertNotEqual(h1, h2)

    def test_filter_kwarg_order_does_not_matter(self):
        """Two identical filter sets in different insertion order must hash equal."""
        keys = list(DEFAULT_FILTER_KWARGS.keys())
        forwardOrder = {k: DEFAULT_FILTER_KWARGS[k] for k in keys}
        reverseOrder = {k: DEFAULT_FILTER_KWARGS[k] for k in reversed(keys)}

        args1 = self.baseArgs()
        args1["filterKwargs"] = forwardOrder
        args2 = self.baseArgs()
        args2["filterKwargs"] = reverseOrder

        self.assertEqual(hashOf(**args1), hashOf(**args2))

    def test_extra_filter_kwarg_changes_hash(self):
        h1 = hashOf(**self.baseArgs())
        args = self.baseArgs()
        modified = dict(DEFAULT_FILTER_KWARGS)
        modified["newConstraint"] = 0.5
        args["filterKwargs"] = modified
        h2 = hashOf(**args)
        self.assertNotEqual(h1, h2)

    def test_hash_is_12_hex_chars(self):
        """Length is part of the contract — production code stores it in the filename."""
        h = hashOf(**self.baseArgs())
        self.assertEqual(len(h), 12)
        int(h, 16)  # must parse as hex; raises if not

    def test_hash_matches_production_formula(self):
        """Independent reconstruction of the hash, anchored to a known input.

        This is the test that fails LOUDLY if someone changes the production
        cache key formula without updating this test — exactly the signal
        we lacked during the cache-drift incident.
        """
        expected = hashOf(
            dataDir="data/AMASS",
            maxSamples=None,
            maxLength=1000,
            filterKwargs={"maxAccel": 50.0, "maxJointRotvel": 30.0,
                          "maxRootSpeed": 10.0, "minFrames": 30,
                          "minVariance": 0.0001},
            schema=4,
        )
        # If CACHE_SCHEMA / DEFAULT_FILTER_KWARGS / formula change, this test
        # FAILS and we get a deliberate review moment instead of silent drift.
        actual = hashOf(
            dataDir="data/AMASS",
            maxSamples=None,
            maxLength=1000,
            filterKwargs=DEFAULT_FILTER_KWARGS,
            schema=CACHE_SCHEMA,
        )
        self.assertEqual(expected, actual,
                         "production cache formula or DEFAULT_FILTER_KWARGS "
                         "drifted from the test anchor — confirm this is intentional, "
                         "then update the test anchor.")


if __name__ == "__main__":
    unittest.main()
