"""Patch the false-positive renames from migrate_camel_to_snake.py.

The bulk migration script over-collected attribute names from external
APIs (unittest.TestCase methods, logging stdlib, etc.) and renamed them
to snake_case despite those being upstream contracts we can't change.

This script does targeted reverse substitutions to restore each
incorrectly-renamed name to its original camelCase form, project-wide.

Plus one one-off acronym fix: encode_texts_t2_m -> encode_texts_t2m.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (incorrect_snake, correct_camel). All from running the bulk migration
# against rename_map.json and cross-checking against stdlib docs.
REVERTS: dict[str, str] = {
    # unittest.TestCase methods — stdlib API
    "assert_almost_equal": "assertAlmostEqual",
    "assert_equal": "assertEqual",
    "assert_false": "assertFalse",
    "assert_greater": "assertGreater",
    "assert_greater_equal": "assertGreaterEqual",
    "assert_in": "assertIn",
    "assert_is": "assertIs",
    "assert_is_instance": "assertIsInstance",
    "assert_is_none": "assertIsNone",
    "assert_is_not_none": "assertIsNotNone",
    "assert_less": "assertLess",
    "assert_less_equal": "assertLessEqual",
    "assert_not_equal": "assertNotEqual",
    "assert_not_in": "assertNotIn",
    "assert_raises": "assertRaises",
    "assert_true": "assertTrue",
    "set_up": "setUp",
    "set_up_class": "setUpClass",
    "skip_test": "skipTest",
    # logging stdlib functions/methods
    "basic_config": "basicConfig",
    "get_logger": "getLogger",
}

# One-off acronym fix: digit_letter mid-name should not be split. The
# migration's heuristic inserted an underscore between `2` and `M` so
# `encodeTextsT2M` became `encode_texts_t2_m` instead of `encode_texts_t2m`.
ACRONYM_FIXES: dict[str, str] = {
    "encode_texts_t2_m": "encode_texts_t2m",
}


def apply_reverts(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    changes = 0
    out = src
    for wrong, right in {**REVERTS, **ACRONYM_FIXES}.items():
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b")
        new_out, n = pattern.subn(right, out)
        out = new_out
        changes += n
    if out != src:
        path.write_text(out, encoding="utf-8", newline="\n")
    return changes


def main() -> None:
    py_files = [
        p for p in ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
        and ".git" not in p.parts
        and ".venv" not in p.parts
        # Don't rewrite the migration / patch scripts themselves while running.
        and p.name not in {
            "migrate_camel_to_snake.py", "patch_external_rename_misfires.py",
        }
    ]
    print(f"Patching {len(py_files)} files ...")
    total = 0
    touched = 0
    for path in py_files:
        n = apply_reverts(path)
        if n:
            touched += 1
            total += n
    print(f"Reverted {total} external/acronym misfires across {touched} files.")


if __name__ == "__main__":
    main()
