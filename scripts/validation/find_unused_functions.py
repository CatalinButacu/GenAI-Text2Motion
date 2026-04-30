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


def collectPyFiles() -> list[Path]:
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

    def __init__(self, filePath: Path) -> None:
        self.filePath = filePath
        self.scopes: list[str] = []
        self.defs: list[dict] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.recordDef(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.recordDef(node)

    def recordDef(self, node) -> None:
        qualname = ".".join(self.scopes + [node.name])
        decorators = [ast.unparse(d) for d in node.decorator_list]
        isMethod = bool(self.scopes)
        self.defs.append({
            "name": node.name,
            "qualname": qualname,
            "lineno": node.lineno,
            "file": str(self.filePath.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "isMethod": isMethod,
            "decorators": decorators,
            "isPrivate": node.name.startswith("_") and not node.name.startswith("__"),
            "isDunder": node.name.startswith("__") and node.name.endswith("__"),
        })

        # Recurse into nested defs / classes
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()


def collectAllDefs(files: list[Path]) -> list[dict]:
    allDefs: list[dict] = []

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
        allDefs.extend(v.defs)

    return allDefs


def buildSourceCorpus(files: list[Path]) -> dict[Path, str]:
    return {f: f.read_text(encoding="utf-8", errors="ignore") for f in files}


def countReferences(name: str, corpus: dict[Path, str], defFile: Path) -> tuple[int, list[str]]:
    """Count occurrences of bare identifier `name` outside the file where it is defined.
    Also returns occurrences inside the same file but on lines other than the def line.
    """
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    externalHits = 0
    sampleFiles: list[str] = []

    for path, text in corpus.items():
        if path == defFile:
            continue

        hits = pattern.findall(text)
        if hits:
            externalHits += len(hits)
            if len(sampleFiles) < 3:
                sampleFiles.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    return externalHits, sampleFiles


def countSelfFileNonDef(name: str, source: str, defLine: int) -> int:
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    total = 0

    for i, line in enumerate(source.splitlines(), start=1):
        if i == defLine:
            continue
        total += len(pattern.findall(line))

    return total


def isAlwaysUsed(d: dict) -> bool:
    name = d["name"]
    if d["isDunder"]:
        return True
    if name in ALWAYS_USED_EXACT:
        return True
    for pref in ALWAYS_USED_PREFIXES:
        if name.startswith(pref):
            return True
    decoStr = " ".join(d["decorators"])
    # pytest fixtures / parametrize / hypothesis / property decorators
    if any(tag in decoStr for tag in ("fixture", "parametrize", "given(", "property",
                                       "abstractmethod", "register", "click.command",
                                       "click.group", "app.command", "app.callback",
                                       "staticmethod", "classmethod")):
        # property/staticmethod/classmethod don't really mark "framework-used"; only
        # treat pytest/registration ones as always-used.
        if any(tag in decoStr for tag in ("fixture", "parametrize", "given(",
                                          "register", "click.command", "click.group",
                                          "app.command", "app.callback")):
            return True
    return False


def main() -> None:
    files = collectPyFiles()
    print(f"Scanning {len(files)} Python files...")

    defs = collectAllDefs(files)
    corpus = buildSourceCorpus(files)
    print(f"Collected {len(defs)} function/method definitions.")

    # Group definitions by name to handle duplicates (e.g., overrides)
    byName: dict[str, list[dict]] = defaultdict(list)
    for d in defs:
        byName[d["name"]].append(d)

    results: list[dict] = []

    for d in defs:
        name = d["name"]
        defFile = PROJECT_ROOT / d["file"]

        externalHits, sampleFiles = countReferences(name, corpus, defFile)
        selfHits = countSelfFileNonDef(name, corpus[defFile], d["lineno"])
        # If multiple defs share the name, the "external" hits may simply be the other defs.
        sameNameDefs = len(byName[name]) - 1

        d["externalHits"] = externalHits
        d["selfHits"] = selfHits
        d["sameNameDefs"] = sameNameDefs
        d["sampleFiles"] = sampleFiles
        d["alwaysUsed"] = isAlwaysUsed(d)

        # Heuristic: unused if no external hits AND no self-file non-def hits
        # (allow same-name overloads to confuse this; we surface that flag separately)
        d["unused"] = (externalHits == 0 and selfHits == 0 and not d["alwaysUsed"])

        results.append(d)

    unused = [d for d in results if d["unused"]]
    print(f"Found {len(unused)} candidate-unused definitions.")

    # Sort: methods of a class first by file/qualname; standalone functions separately
    unused.sort(key=lambda d: (d["file"], d["lineno"]))

    outPath = PROJECT_ROOT / "scripts" / "validation" / "unused_functions.json"
    outPath.write_text(json.dumps({
        "totalDefs": len(defs),
        "unusedCount": len(unused),
        "unused": unused,
        "all": results,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {outPath}")

    # Also print a compact markdown table to stdout
    print("\n## Unused function candidates\n")
    print("| File | Line | Qualname | Kind | Decorators |")
    print("|------|------|----------|------|------------|")
    for d in unused:
        kind = "method" if d["isMethod"] else "function"
        if d["isPrivate"]:
            kind = "private " + kind
        deco = ",".join(d["decorators"]) if d["decorators"] else ""
        print(f"| {d['file']} | {d['lineno']} | {d['qualname']} | {kind} | {deco} |")


if __name__ == "__main__":
    main()
