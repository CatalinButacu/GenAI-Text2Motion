"""Style rule regression tests.

Runs the AST-based style checker (scripts/validation/check_style.py) and
enforces two guarantees:

  1. Rules 4 and 6 must have ZERO violations at all times.
     These are fully fixable structural rules (no inline imports, no silent
     try/except). Any new violation is a clear regression.

  2. Rules 1 and 2 track a frozen violation BASELINE. The test fails if the
     count INCREASES (new violations introduced). It passes if the count stays
     equal or DECREASES (migration progress). Lower the baseline numbers as
     legacy code is renamed to PEP 8 snake_case.

Polarity note (2026-05-22):
    Rule 2 used to flag snake_case names (when the project was camelCase).
    After the project-wide migration the polarity is inverted: Rule 2
    now flags camelCase names that have not yet been renamed. The
    surviving 17 are residual debt to chase to zero over time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
from scripts.validation.check_style import StyleReport, run

SCAN_TARGETS = [ROOT / "src", ROOT / "main.py"]

# Frozen baselines -- reduce these as migration continues; never increase them.
# After the 2026-05-22 snake_case migration the entire src/ + main.py is at
# zero camelCase. All four rules are now hard-zero. New camelCase or
# underscore-prefixed names are immediate regressions.
BASELINE: dict[int, int] = {
    1: 0,    # underscore-prefixed names -- must stay zero
    2: 0,    # camelCase names -- must stay zero (PEP 8 is now the standard)
    4: 0,    # inline imports -- must stay zero
    6: 0,    # silent try/except -- must stay zero
}

HARD_ZERO_RULES: set[int] = {1, 2, 4, 6}

@pytest.fixture(scope="module")
def style_report() -> StyleReport:
    return run(targets=SCAN_TARGETS)

def test_rule4_no_inline_imports(style_report: StyleReport) -> None:
    """Rule 4: no imports inside functions or class bodies."""
    violations = [v for v in style_report.violations if v.rule == 4]
    msgs = "\n".join(str(v) for v in violations)
    assert len(violations) == 0, f"Rule 4 violations (must be zero):\n{msgs}"

def test_rule6_no_silent_exceptions(style_report: StyleReport) -> None:
    """Rule 6: no broad try/except that silently swallows exceptions."""
    violations = [v for v in style_report.violations if v.rule == 6]
    msgs = "\n".join(str(v) for v in violations)
    assert len(violations) == 0, f"Rule 6 violations (must be zero):\n{msgs}"

def test_rule1_underscore_not_increasing(style_report: StyleReport) -> None:
    """Rule 1: underscore-prefixed names must not increase beyond baseline.

    After the snake_case migration we hit zero here; treat any new
    underscore-prefixed name as a regression.
    """
    violations = [v for v in style_report.violations if v.rule == 1]
    baseline = BASELINE[1]
    count = len(violations)
    details = "\n".join(str(v) for v in violations)
    assert count <= baseline, (
        f"Rule 1 regressions: {count} violations but baseline is {baseline}.\n"
        f"New underscore-prefixed names introduced -- remove the leading underscore:\n{details}"
    )

def test_rule2_camelcase_not_increasing(style_report: StyleReport) -> None:
    """Rule 2: camelCase names must not increase beyond baseline.

    After 2026-05-22 the project is snake_case; the remaining baseline is
    legacy debt being chased to zero. Any new camelCase name fails the test.
    """
    violations = [v for v in style_report.violations if v.rule == 2]
    baseline = BASELINE[2]
    count = len(violations)
    details = "\n".join(str(v) for v in violations[:20])
    suffix = f"\n  ... and {count - 20} more" if count > 20 else ""
    assert count <= baseline, (
        f"Rule 2 regressions: {count} violations but baseline is {baseline}.\n"
        f"New camelCase names introduced -- rename to snake_case:\n{details}{suffix}"
    )
    if count < baseline:
        pytest.skip(
            f"Rule 2 improved: {count} violations < baseline {baseline}. "
            f"Update BASELINE[2] = {count} in this file."
        )
