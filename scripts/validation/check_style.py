#!/usr/bin/env python
"""Style rule validator — checks Rules 1, 2, 4, 6 from the project style guide.

Rules enforced
--------------
  Rule 1 — No underscore-prefixed names (non-dunder functions/variables).
  Rule 2 — camelCase for functions and local variables (no snake_case).
  Rule 4 — All imports at top of file (no inline imports).
  Rule 6 — No excessive try/except that swallows exceptions silently.

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

    def byRule(self) -> dict[int, list[Violation]]:
        groups: dict[int, list[Violation]] = {}
        for v in self.violations:
            groups.setdefault(v.rule, []).append(v)
        return groups

    @property
    def count(self) -> int:
        return len(self.violations)


def isPrivateName(name: str) -> bool:
    """True if name starts with _ but is not a dunder."""
    return name.startswith("_") and not DUNDER_RE.match(name)


def isSnakeName(name: str) -> bool:
    """True if name looks like snake_case (not camelCase, not UPPER_CONST, not dunder)."""
    if DUNDER_RE.match(name):
        return False
    if UPPER_CONST_RE.match(name):
        return False
    return bool(SNAKE_RE.match(name))


def checkRule1(tree: ast.AST, filepath: Path, report: StyleReport) -> None:
    """Rule 1: No underscore-prefixed function or variable names (non-dunder)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isPrivateName(node.name):
                report.add(1, filepath, node.lineno,
                            f"function `{node.name}` starts with `_` (rename to camelCase)")

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isPrivateName(target.id):
                    report.add(1, filepath, node.lineno,
                                f"variable `{target.id}` starts with `_`")

        elif isinstance(node, (ast.AnnAssign,)):
            if isinstance(node.target, ast.Name) and isPrivateName(node.target.id):
                report.add(1, filepath, node.lineno,
                            f"variable `{node.target.id}` starts with `_`")


def checkRule2(tree: ast.AST, filepath: Path, report: StyleReport) -> None:
    """Rule 2: camelCase for function definitions and local variable assignments."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not DUNDER_RE.match(node.name) and isSnakeName(node.name):
                report.add(2, filepath, node.lineno,
                            f"function `{node.name}` is snake_case (use camelCase)")

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isSnakeName(target.id):
                    report.add(2, filepath, node.lineno,
                                f"variable `{target.id}` is snake_case (use camelCase)")

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and isSnakeName(node.target.id):
                report.add(2, filepath, node.lineno,
                            f"variable `{node.target.id}` is snake_case (use camelCase)")


def checkRule4(tree: ast.AST, filepath: Path, report: StyleReport) -> None:
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


def checkRule6(tree: ast.AST, filepath: Path, report: StyleReport) -> None:
    """Rule 6: No try/except that silently swallows exceptions (bare pass handlers)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue

        for handler in node.handlers:
            body = handler.body
            isSilent = (
                len(body) == 1
                and isinstance(body[0], ast.Pass)
            )
            isBroad = (
                handler.type is None
                or (isinstance(handler.type, ast.Name) and handler.type.id == "Exception")
                or (isinstance(handler.type, ast.Attribute) and handler.type.attr == "Exception")
            )

            if isSilent and isBroad:
                report.add(6, filepath, handler.lineno,
                            "broad `except` with only `pass` — swallows all exceptions silently")


def checkFile(filepath: Path, report: StyleReport) -> None:
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        report.add(0, filepath, e.lineno or 0, f"SyntaxError: {e.msg}")
        return

    checkRule1(tree, filepath, report)
    checkRule2(tree, filepath, report)
    checkRule4(tree, filepath, report)
    checkRule6(tree, filepath, report)


def collectFiles(targets: list[Path]) -> list[Path]:
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


def run(targets: list[Path] | None = None, asJson: bool = False) -> StyleReport:
    report = StyleReport()
    files = collectFiles(targets or SCAN_TARGETS)

    for filepath in files:
        checkFile(filepath, report)

    if asJson:
        data = [
            {"rule": v.rule, "file": v.filepath, "line": v.line, "message": v.message}
            for v in report.violations
        ]
        print(json.dumps(data, indent=2))
        return report

    groups = report.byRule()
    ruleNames = {
        1: "No underscore-prefixed names",
        2: "camelCase for functions and variables",
        4: "All imports at top of file",
        6: "No excessive try/except (silent pass handlers)",
    }

    if not report.violations:
        print("Style check passed — 0 violations.")
        return report

    print(f"\nStyle violations found: {report.count} total\n")

    for ruleNum in sorted(groups):
        vs = groups[ruleNum]
        print(f"Rule {ruleNum} — {ruleNames.get(ruleNum, '?')} ({len(vs)} violations)")
        for v in vs:
            print(f"  {v.filepath}:{v.line}  {v.message}")
        print()

    return report


if __name__ == "__main__":
    asJson = "--json" in sys.argv
    isStrict = "--strict" in sys.argv

    report = run(asJson=asJson)

    if report.count > 0:
        if not asJson:
            print(f"Total: {report.count} violation(s). Fix before committing.")
        sys.exit(1)
