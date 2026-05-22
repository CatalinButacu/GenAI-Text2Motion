"""Validate the synthesized planner dataset is internally consistent.

Generates a small sample with the synthesis script and asserts:

  1. Every example is a well-formed JSON object with the expected schema.
  2. Every ``until`` field parses cleanly via
     :func:`src.modules.agent.conditions.parse_condition`. This is the
     bridge contract -- the LM is trained to emit ``until`` strings that
     the runtime can evaluate, and if the dataset's strings don't satisfy
     this then the LM will learn an unparseable distribution.
  3. The dataset uses every grammar branch (atomic, duration, distance,
     rotation) so the LM sees the full output language during training.

The synthesis script is invoked in-process so the test is reproducible
without shelling out and without depending on a persisted dataset file.
"""
from __future__ import annotations

import json
import unittest
from io import StringIO

from src.modules.agent.conditions import parse_condition


class TestPlannerDatasetSchema(unittest.TestCase):

    def setUp(self) -> None:
        # Generate 200 examples in-process. Re-importing the script's
        # _sample_example gives us full control over the RNG and avoids
        # touching disk.
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "synth",
            Path(__file__).resolve().parents[1] / "scripts" / "data" / "synthesize_planner_data.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import random as _stdlib_random
        rng = _stdlib_random.Random(0)
        self.examples = [mod.sample_example(rng) for _ in range(200)]

    def test_schema_shape(self):
        for ex in self.examples:
            self.assertIn("instruction", ex)
            self.assertIn("actions", ex)
            self.assertIsInstance(ex["instruction"], str)
            self.assertGreater(len(ex["instruction"]), 0)
            self.assertIsInstance(ex["actions"], list)
            self.assertGreater(len(ex["actions"]), 0)

            for act in ex["actions"]:
                self.assertIn("action", act)
                self.assertIn("until", act)
                self.assertIsInstance(act["action"], str)
                self.assertIsInstance(act["until"], str)

    def test_every_until_parses(self):
        """The core bridge invariant: synthesised data must speak the same
        grammar the runtime condition parser accepts.
        """

        for ex in self.examples:
            for act in ex["actions"]:
                try:
                    parse_condition(act["until"])
                except ValueError as e:
                    self.fail(
                        f"synthesised until-string {act['until']!r} "
                        f"failed runtime parse: {e}\nfull example: {ex}"
                    )

    def test_coverage_all_grammar_branches(self):
        """The training set must exercise every condition branch or the LM
        will be biased away from rare predicates at inference time.
        """
        seen_kinds: set[str] = set()

        for ex in self.examples:
            for act in ex["actions"]:
                u = act["until"]
                if u.startswith("duration("):
                    seen_kinds.add("duration")
                elif u.startswith("distance(") and "<" in u:
                    seen_kinds.add("distance_lt")
                elif u.startswith("distance(") and ">" in u:
                    seen_kinds.add("distance_gt")
                elif u.startswith("rotated("):
                    seen_kinds.add("rotated")
                elif u == "completed":
                    seen_kinds.add("completed")
        # All five branches must appear in 200 examples
        for branch in ["duration", "distance_lt", "distance_gt", "rotated", "completed"]:
            self.assertIn(
                branch, seen_kinds,
                f"branch {branch!r} not seen in 200 examples; LM training "
                "will be biased. Increase sample count or rebalance the "
                "structure weights in synthesize_planner_data.py.",
            )

    def test_round_trip_through_jsonl(self):
        """Lines written as JSONL must round-trip cleanly (no embedded
        newlines, trailing commas, etc.)."""
        buf = StringIO()

        for ex in self.examples[:50]:
            buf.write(json.dumps(ex) + "\n")
        buf.seek(0)

        for line in buf:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            self.assertEqual(set(obj.keys()), {"instruction", "actions"})


if __name__ == "__main__":
    unittest.main()
