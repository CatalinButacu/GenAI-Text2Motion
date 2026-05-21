#!/usr/bin/env python
"""Style rule validator — checks Rules 1, 2, 4, 6 from the project style guide.

Rules enforced
--------------
  Rule 1 — No underscore-prefixed names (non-dunder functions/variables).
  Rule 2 — snake_case for functions and local variables (PEP 8). camelCase
           remaining is migration debt -- ratchet down toward zero.
  Rule 4 — All imports at top of file (no inline imports).
  Rule 6 — No excessive try/except that swallows exceptions silently.

Note: as of the 2026-05-22 snake_case migration the project follows PEP 8
naming. Rule 2's polarity is inverted vs the pre-migration era; camelCase
violations are tracked by a frozen baseline that should only decrease.

Usage
-----
  python scripts/validation/check_style.py          # check src/ + main.py
  python scripts/validation/check_style.py --strict # non-zero exit on any violation
  python scripts/validation/check_style.py --json   # machine-readable output

Exit codes
----------
  0  no violations
  1  violations found
"""
from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

SCAN_TARGETS: list[Path] = [
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "main.py",
]

EXCLUDE_DIRS: set[str] = {"arctic-master", "__pycache__", ".git", "vendor"}
EXCLUDE_FILES: set[str] = set()

SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
CAMEL_RE = re.compile(r"^[a-z][a-z0-9]*([A-Z][a-z0-9]*)+$")
DUNDER_RE = re.compile(r"^__\w+__$")
UPPER_CONST_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class Violation:
    rule: int
    filepath: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"  Rule {self.rule}  {self.filepath}:{self.line}  {self.message}"


@dataclass
class StyleReport:
    violations: list[Violation] = field(default_factory=list)

    def add(self, rule: int, filepath: Path, line: int, msg: str) -> None:
        self.violations.append(Violation(rule, str(filepath.relative_to(ROOT)), line, msg))

    def by_rule(self) -> dict[int, list[Violation]]:
        groups: dict[int, list[Violation]] = {}
        for v in self.violations:
            groups.setdefault(v.rule, []).append(v)
        return groups

    @property
    def count(self) -> int:
        return len(self.violations)


def is_private_name(name: str) -> bool:
    """True if name starts with _ but is not a dunder."""
    return name.startswith("_") and not DUNDER_RE.match(name)


def is_snake_name(name: str) -> bool:
    """True if name looks like snake_case (not camelCase, not UPPER_CONST, not dunder)."""
    if DUNDER_RE.match(name):
        return False
    if UPPER_CONST_RE.match(name):
        return False
    return bool(SNAKE_RE.match(name))


def is_camel_name(name: str) -> bool:
    """True if name looks like camelCase: lowercase first + at least one uppercase.

    PascalCase class names (`SomeClass`) and ALL_CAPS constants (`MOTION_FPS`)
    return False.
    """
    if DUNDER_RE.match(name):
        return False
    if UPPER_CONST_RE.match(name):
        return False
    return bool(CAMEL_RE.match(name))


def check_rule1(tree: ast.AST, filepath: Path, report: StyleReport) -> None:
    """Rule 1: No underscore-prefixed function or variable names (non-dunder)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_private_name(node.name):
                report.add(1, filepath, node.lineno,
                            f"function `{node.name}` starts with `_` (rename to camelCase)")

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and is_private_name(target.id):
                    report.add(1, filepath, node.lineno,
                                f"variable `{target.id}` starts with `_`")

        elif isinstance(node, (ast.AnnAssign,)):
            if isinstance(node.target, ast.Name) and is_private_name(node.target.id):
                report.add(1, filepath, node.lineno,
                            f"variable `{node.target.id}` starts with `_`")


def check_rule2(tree: ast.AST, filepath: Path, report: StyleReport) -> None:
    """Rule 2: snake_case for function definitions and local variable assignments.

    POLARITY INVERTED on 2026-05-22 after the project-wide snake_case migration.
    Pre-migration: flagged snake_case names. Post-migration: flags camelCase
    names. The frozen baseline in tests/test_style_rules.py counts remaining
    camelCase debt; it should only ever decrease.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not DUNDER_RE.match(node.name) and is_camel_name(node.name):
                report.add(2, filepath, node.lineno,
                            f"function `{node.name}` is camelCase (use snake_case)")

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and is_camel_name(target.id):
                    report.add(2, filepath, node.lineno,
                                f"variable `{target.id}` is camelCase (use snake_case)")

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and is_camel_name(node.target.id):
                report.add(2, filepath, node.lineno,
                            f"variable `{node.target.id}` is camelCase (use snake_case)")


def check_rule4(tree: ast.AST, filepath: Path, report: StyleReport) -> None:
    """Rule 4: No imports inside functions, classes, or conditional blocks."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                try:
                    src = ast.unparse(child)
                except Exception:
                    src = "<import>"
                report.add(4, filepath, child.lineno,
                            f"inline import inside `{node.name}`: {src}")


def check_rule6(tree: ast.AST, filepath: Path, report: StyleReport) -> None:
    """Rule 6: No try/except that silently swallows exceptions (bare pass handlers)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue

        for handler in node.handlers:
            body = handler.body
            is_silent = (
                len(body) == 1
                and isinstance(body[0], ast.Pass)
            )
            is_broad = (
                handler.type is None
                or (isinstance(handler.type, ast.Name) and handler.type.id == "Exception")
                or (isinstance(handler.type, ast.Attribute) and handler.type.attr == "Exception")
            )

            if is_silent and is_broad:
                report.add(6, filepath, handler.lineno,
                            "broad `except` with only `pass` — swallows all exceptions silently")


def check_file(filepath: Path, report: StyleReport) -> None:
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        report.add(0, filepath, e.lineno or 0, f"SyntaxError: {e.msg}")
        return

    check_rule1(tree, filepath, report)
    check_rule2(tree, filepath, report)
    check_rule4(tree, filepath, report)
    check_rule6(tree, filepath, report)


def collect_files(targets: list[Path]) -> list[Path]:
    files: list[Path] = []

    for target in targets:
        if target.is_file() and target.suffix == ".py":
            files.append(target)
        elif target.is_dir():
            for py in sorted(target.rglob("*.py")):
                if any(ex in py.parts for ex in EXCLUDE_DIRS):
                    continue
                if py.name in EXCLUDE_FILES:
                    continue
                files.append(py)

    return files


def run(targets: list[Path] | None = None, as_json: bool = False) -> StyleReport:
    report = StyleReport()
    files = collect_files(targets or SCAN_TARGETS)

    for filepath in files:
        check_file(filepath, report)

    if as_json:
        data = [
            {"rule": v.rule, "file": v.filepath, "line": v.line, "message": v.message}
            for v in report.violations
        ]
        print(json.dumps(data, indent=2))
        return report

    groups = report.by_rule()
    rule_names = {
        1: "No underscore-prefixed names",
        2: "camelCase for functions and variables",
        4: "All imports at top of file",
        6: "No excessive try/except (silent pass handlers)",
    }

    if not report.violations:
        print("Style check passed — 0 violations.")
        return report

    print(f"\nStyle violations found: {report.count} total\n")

    for rule_num in sorted(groups):
        vs = groups[rule_num]
        print(f"Rule {rule_num} — {rule_names.get(rule_num, '?')} ({len(vs)} violations)")
        for v in vs:
            print(f"  {v.filepath}:{v.line}  {v.message}")
        print()

    return report


if __name__ == "__main__":
    as_json = "--json" in sys.argv
    is_strict = "--strict" in sys.argv

    report = run(as_json=as_json)

    if report.count > 0:
        if not as_json:
            print(f"Total: {report.count} violation(s). Fix before committing.")
        sys.exit(1)
