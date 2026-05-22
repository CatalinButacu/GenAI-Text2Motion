"""Parse-side tests for ActionPlanner -- runs without a trained LM checkpoint.

The planner has two failure surfaces: LM generates malformed JSON, or LM
generates valid JSON whose ``until`` field doesn't conform to the runtime
grammar. Both must raise loudly so the demo never feeds garbage to the
motion stack.

This test directly exercises :meth:`ActionPlanner._parse` by constructing
the parse-only fixture in-process and feeding it crafted completions.
"""
from __future__ import annotations

import unittest

from src.modules.agent.planner import ActionPlanner


class _ParseOnly:
    """Minimal stand-in for ActionPlanner that exposes only ._parse.

    Lets us test the parser without the transformers dependency or a real
    checkpoint file. Constructed with object.__new__ to skip __init__.
    """

    def __init__(self) -> None:
        # Borrow the bound method without going through __init__
        self._parse = ActionPlanner._parse.__get__(self, ActionPlanner)


class TestActionPlannerParse(unittest.TestCase):

    def setUp(self) -> None:
        self.parser = _ParseOnly()

    def test_well_formed_single_action(self):
        out = self.parser._parse('[{"action": "walk forward", "until": "duration(40)"}]')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].action_text, "walk forward")
        self.assertEqual(out[0].until.source, "duration(40)")

    def test_well_formed_multi_action(self):
        completion = (
            '[{"action": "walk forward", "until": "distance(tree) < 1.0"}, '
            '{"action": "turn", "until": "rotated(90)"}, '
            '{"action": "wave", "until": "completed"}]'
        )
        out = self.parser._parse(completion)
        self.assertEqual(len(out), 3)
        self.assertEqual(
            [a.action_text for a in out],
            ["walk forward", "turn", "wave"],
        )

    def test_trailing_garbage_is_clipped(self):
        """The LM sometimes generates after the closing bracket; the parser
        clips to the first ']' and retries."""
        out = self.parser._parse(
            '[{"action": "wave", "until": "completed"}]\nInstruction: next prompt...'
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].action_text, "wave")

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            self.parser._parse('not even json at all')

    def test_non_list_top_level_raises(self):
        with self.assertRaises(ValueError):
            self.parser._parse('{"action": "walk", "until": "completed"}')

    def test_missing_required_keys_raises(self):
        with self.assertRaises(ValueError):
            self.parser._parse('[{"action": "walk"}]')  # missing until

        with self.assertRaises(ValueError):
            self.parser._parse('[{"until": "completed"}]')  # missing action

    def test_unparseable_until_raises(self):
        """Even if JSON is valid, an until-string outside the runtime grammar
        must fail loud at parse time, not silently at runner-eval time."""
        with self.assertRaises(ValueError):
            self.parser._parse('[{"action": "walk", "until": "until reach tree"}]')

        with self.assertRaises(ValueError):
            self.parser._parse('[{"action": "walk", "until": "while moving"}]')

    def test_action_dict_not_dict_raises(self):
        with self.assertRaises(ValueError):
            self.parser._parse('["walk forward"]')  # string in list, not dict


if __name__ == "__main__":
    unittest.main()
