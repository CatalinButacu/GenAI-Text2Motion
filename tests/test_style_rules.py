"""Style rule regression tests.

Runs the AST-based style checker (scripts/validation/check_style.py) and
enforces two guarantees:

  1. Rules 4 and 6 must have ZERO violations at all times.
     These are fully fixable structural rules (no inline imports, no silent
     try/except). Any new violation is a clear regression.

  2. Rules 1 and 2 track a frozen violation BASELINE. The test fails if the
     count INCREASES (new violations introduced). It passes if the count stays
     equal or DECREASES (migration progress). Lower the baseline numbers as
     legacy code is renamed.

BASELINE values reflect the state after the initial migration pass. Reduce
each number as more files are renamed to camelCase / underscore-free style.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
from scripts.validation.check_style import StyleReport, run

SCAN_TARGETS = [ROOT / "src", ROOT / "main.py"]

# Frozen baselines — reduce these as migration continues; never increase them.
BASELINE: dict[int, int] = {
    1: 19,   # underscore-prefixed names (legacy; being renamed)
    2: 474,  # snake_case names (legacy; being renamed)
    4: 0,    # inline imports — must stay zero
    6: 0,    # silent try/except — must stay zero
}

HARD_ZERO_RULES: set[int] = {4, 6}

@pytest.fixture(scope="module")
def styleReport() -> StyleReport:
    return run(targets=SCAN_TARGETS)

def test_rule4_no_inline_imports(styleReport: StyleReport) -> None:
    """Rule 4: no imports inside functions or class bodies."""
    violations = [v for v in styleReport.violations if v.rule == 4]
    msgs = "\n".join(str(v) for v in violations)
    assert len(violations) == 0, f"Rule 4 violations (must be zero):\n{msgs}"

def test_rule6_no_silent_exceptions(styleReport: StyleReport) -> None:
    """Rule 6: no broad try/except that silently swallows exceptions."""
    violations = [v for v in styleReport.violations if v.rule == 6]
    msgs = "\n".join(str(v) for v in violations)
    assert len(violations) == 0, f"Rule 6 violations (must be zero):\n{msgs}"

def test_rule1_underscore_not_increasing(styleReport: StyleReport) -> None:
    """Rule 1: underscore-prefixed names must not increase beyond baseline."""
    violations = [v for v in styleReport.violations if v.rule == 1]
    baseline = BASELINE[1]
    count = len(violations)
    details = "\n".join(str(v) for v in violations)
    assert count <= baseline, (
        f"Rule 1 regressions: {count} violations but baseline is {baseline}.\n"
        f"New violations introduced — rename them to camelCase:\n{details}"
    )
    if count < baseline:
        pytest.skip(
            f"Rule 1 improved: {count} violations < baseline {baseline}. "
            f"Update BASELINE[1] = {count} in this file."
        )

def test_rule2_snakecase_not_increasing(styleReport: StyleReport) -> None:
    """Rule 2: snake_case names must not increase beyond baseline."""
    violations = [v for v in styleReport.violations if v.rule == 2]
    baseline = BASELINE[2]
    count = len(violations)
    details = "\n".join(str(v) for v in violations[:20])
    suffix = f"\n  ... and {count - 20} more" if count > 20 else ""
    assert count <= baseline, (
        f"Rule 2 regressions: {count} violations but baseline is {baseline}.\n"
        f"New snake_case names introduced — rename them:\n{details}{suffix}"
    )
    if count < baseline:
        pytest.skip(
            f"Rule 2 improved: {count} violations < baseline {baseline}. "
            f"Update BASELINE[2] = {count} in this file."
        )
