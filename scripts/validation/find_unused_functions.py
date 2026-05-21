"""
One-shot analysis: list every function defined in the project, count references,
and flag candidates for removal.

A function is reported as "unused" if its bare name appears in NO source file
outside its own definition. We treat dunder methods, special pytest names,
CLI entry points, and a few framework hooks as "always-used".

Output: stdout markdown table + JSON file at scripts/validation/unused_functions.json
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ["src", "scripts", "tests"]
TOP_LEVEL_FILES = ["main.py"]

EXCLUDE_DIRS = {"__pycache__", ".venv", "venv", ".git", "node_modules",
                "checkpoints", "outputs", "doc", "notebooks", ".pytest_cache",
                ".ruff_cache", ".ropeproject"}
# Exclude only the TOP-LEVEL data dir (datasets), not src/data/ (real code).
EXCLUDE_TOPLEVEL_DIRS = {"data"}

# Names we never report as "unused" even if no caller is found.
ALWAYS_USED_PREFIXES = ("test_", "Test")
ALWAYS_USED_EXACT = {
    "main", "setUp", "tearDown", "setUpClass", "tearDownClass",
    "setup_module", "teardown_module", "pytest_collection_modifyitems",
    "conftest",
}


def collect_py_files() -> list[Path]:
    files: list[Path] = []

    for top in TOP_LEVEL_FILES:
        path = PROJECT_ROOT / top
        if path.exists():
            files.append(path)

    for sub in SOURCE_DIRS:
        base = PROJECT_ROOT / sub
        if not base.exists():
            continue

        for p in base.rglob("*.py"):
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            try:
                rel = p.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = p
            if rel.parts and rel.parts[0] in EXCLUDE_TOPLEVEL_DIRS:
                continue
            files.append(p)

    return files


class DefVisitor(ast.NodeVisitor):
    """Collect every function/method defined in a file with its qualname and decorators."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.scopes: list[str] = []
        self.defs: list[dict] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.record_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.record_def(node)

    def record_def(self, node) -> None:
        qualname = ".".join(self.scopes + [node.name])
        decorators = [ast.unparse(d) for d in node.decorator_list]
        is_method = bool(self.scopes)
        self.defs.append({
            "name": node.name,
            "qualname": qualname,
            "lineno": node.lineno,
            "file": str(self.file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "is_method": is_method,
            "decorators": decorators,
            "isPrivate": node.name.startswith("_") and not node.name.startswith("__"),
            "isDunder": node.name.startswith("__") and node.name.endswith("__"),
        })

        # Recurse into nested defs / classes
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()


def collect_all_defs(files: list[Path]) -> list[dict]:
    all_defs: list[dict] = []

    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src, filename=str(f))
        except SyntaxError:
            continue

        v = DefVisitor(f)
        v.visit(tree)
        all_defs.extend(v.defs)

    return all_defs


def build_source_corpus(files: list[Path]) -> dict[Path, str]:
    return {f: f.read_text(encoding="utf-8", errors="ignore") for f in files}


def count_references(name: str, corpus: dict[Path, str], def_file: Path) -> tuple[int, list[str]]:
    """Count occurrences of bare identifier `name` outside the file where it is defined.
    Also returns occurrences inside the same file but on lines other than the def line.
    """
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    external_hits = 0
    sample_files: list[str] = []

    for path, text in corpus.items():
        if path == def_file:
            continue

        hits = pattern.findall(text)
        if hits:
            external_hits += len(hits)
            if len(sample_files) < 3:
                sample_files.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    return external_hits, sample_files


def count_self_file_non_def(name: str, source: str, def_line: int) -> int:
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    total = 0

    for i, line in enumerate(source.splitlines(), start=1):
        if i == def_line:
            continue
        total += len(pattern.findall(line))

    return total


def is_always_used(d: dict) -> bool:
    name = d["name"]
    if d["isDunder"]:
        return True
    if name in ALWAYS_USED_EXACT:
        return True
    for pref in ALWAYS_USED_PREFIXES:
        if name.startswith(pref):
            return True
    deco_str = " ".join(d["decorators"])
    # pytest fixtures / parametrize / hypothesis / property decorators
    if any(tag in deco_str for tag in ("fixture", "parametrize", "given(", "property",
                                       "abstractmethod", "register", "click.command",
                                       "click.group", "app.command", "app.callback",
                                       "staticmethod", "classmethod")):
        # property/staticmethod/classmethod don't really mark "framework-used"; only
        # treat pytest/registration ones as always-used.
        if any(tag in deco_str for tag in ("fixture", "parametrize", "given(",
                                          "register", "click.command", "click.group",
                                          "app.command", "app.callback")):
            return True
    return False


def main() -> None:
    files = collect_py_files()
    print(f"Scanning {len(files)} Python files...")

    defs = collect_all_defs(files)
    corpus = build_source_corpus(files)
    print(f"Collected {len(defs)} function/method definitions.")

    # Group definitions by name to handle duplicates (e.g., overrides)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for d in defs:
        by_name[d["name"]].append(d)

    results: list[dict] = []

    for d in defs:
        name = d["name"]
        def_file = PROJECT_ROOT / d["file"]

        external_hits, sample_files = count_references(name, corpus, def_file)
        self_hits = count_self_file_non_def(name, corpus[def_file], d["lineno"])
        # If multiple defs share the name, the "external" hits may simply be the other defs.
        same_name_defs = len(by_name[name]) - 1

        d["external_hits"] = external_hits
        d["self_hits"] = self_hits
        d["same_name_defs"] = same_name_defs
        d["sample_files"] = sample_files
        d["alwaysUsed"] = is_always_used(d)

        # Heuristic: unused if no external hits AND no self-file non-def hits
        # (allow same-name overloads to confuse this; we surface that flag separately)
        d["unused"] = (external_hits == 0 and self_hits == 0 and not d["alwaysUsed"])

        results.append(d)

    unused = [d for d in results if d["unused"]]
    print(f"Found {len(unused)} candidate-unused definitions.")

    # Sort: methods of a class first by file/qualname; standalone functions separately
    unused.sort(key=lambda d: (d["file"], d["lineno"]))

    out_path = PROJECT_ROOT / "scripts" / "validation" / "unused_functions.json"
    out_path.write_text(json.dumps({
        "totalDefs": len(defs),
        "unusedCount": len(unused),
        "unused": unused,
        "all": results,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Also print a compact markdown table to stdout
    print("\n## Unused function candidates\n")
    print("| File | Line | Qualname | Kind | Decorators |")
    print("|------|------|----------|------|------------|")
    for d in unused:
        kind = "method" if d["is_method"] else "function"
        if d["isPrivate"]:
            kind = "private " + kind
        deco = ",".join(d["decorators"]) if d["decorators"] else ""
        print(f"| {d['file']} | {d['lineno']} | {d['qualname']} | {kind} | {deco} |")


if __name__ == "__main__":
    main()
