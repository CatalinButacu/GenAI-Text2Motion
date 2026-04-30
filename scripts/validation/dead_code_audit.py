"""
Deeper dead-code audit:
  1. Orphan modules: live .py files that are never imported by any other file
     AND don't look like entry points / CLI scripts / test files.
  2. Low-use functions: standalone functions with <= maxRefs external references,
     listed with file:line of every caller — for hand inspection.

Output: stdout report + JSON at scripts/validation/dead_code_audit.json
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
EXCLUDE_DIRS = {"__pycache__", ".venv", ".git", "node_modules", "checkpoints",
                "outputs", "doc", "notebooks", ".pytest_cache", ".ruff_cache",
                ".ropeproject"}
EXCLUDE_TOPLEVEL = {"data"}

# Things that look like entry points / scripts; exempt from "orphan module" report.
ENTRY_POINT_PATTERNS = [
    re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]"),
]


def collectFiles() -> list[Path]:
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
            rel = p.relative_to(PROJECT_ROOT)
            if rel.parts and rel.parts[0] in EXCLUDE_TOPLEVEL:
                continue
            files.append(p)

    return files


def fileToModule(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip .py

    return ".".join(parts)


def collectImports(text: str, fileMod: str) -> set[str]:
    """Return all module-name strings imported by this file (best-effort).
    Resolves relative imports against fileMod (the importing file's dotted name)."""
    imports: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return imports

    parts = fileMod.split(".") if fileMod else []
    pkgParts = parts[:-1]  # the package this file lives in

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
                # Also record the imported names (for `import a.b.c` add a, a.b, a.b.c)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            base: list[str] = []
            if level > 0:
                # relative — strip `level` parts off pkgParts
                base = pkgParts[: max(0, len(pkgParts) - level + 1)]
            mod = node.module or ""
            full = ".".join([*base, mod]) if mod else ".".join(base)
            full = full.strip(".")
            if full:
                imports.add(full)
            # Also include each `from X import Y` name (Y could be a submodule)
            for n in node.names:
                if full:
                    imports.add(f"{full}.{n.name}")
                else:
                    imports.add(n.name)

    return imports


def isEntryFile(path: Path, text: str) -> bool:
    if path.name == "main.py":
        return True
    if path.parts[-1].startswith("conftest"):
        return True
    if "tests" in path.parts:
        return True
    if path.parts[-1].startswith("test_"):
        return True
    if "scripts" in path.parts:
        return True  # any script is a potential entry
    for pat in ENTRY_POINT_PATTERNS:
        if pat.search(text):
            return True

    return False


class DefVisitor(ast.NodeVisitor):
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
        self.defs.append({
            "name": node.name,
            "qualname": qualname,
            "lineno": node.lineno,
            "file": str(self.filePath.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "isMethod": bool(self.scopes),
            "decorators": decorators,
            "isPrivate": node.name.startswith("_") and not node.name.startswith("__"),
            "isDunder": node.name.startswith("__") and node.name.endswith("__"),
        })
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()


def buildCorpus(files: list[Path]) -> dict[Path, str]:
    return {f: f.read_text(encoding="utf-8", errors="ignore") for f in files}


def findReferences(name: str, corpus: dict[Path, str], defFile: Path) -> list[tuple[str, int, str]]:
    """List every external reference site as (file, line, line_text)."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    out: list[tuple[str, int, str]] = []

    for path, text in corpus.items():
        if path == defFile:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                out.append((rel, i, line.strip()))

    return out


def isAlwaysUsed(d: dict) -> bool:
    name = d["name"]
    if d["isDunder"]:
        return True
    if name in {"main", "setUp", "tearDown", "setUpClass", "tearDownClass"}:
        return True
    if name.startswith("test_") or name.startswith("Test"):
        return True
    decoStr = " ".join(d["decorators"])
    if any(tag in decoStr for tag in ("fixture", "parametrize", "given(", "register",
                                       "click.command", "click.group", "app.command",
                                       "app.callback")):
        return True
    # ast.NodeVisitor dispatch
    if name.startswith("visit_") and any(
        "Visitor" in dec or "ast" in dec for dec in d["decorators"]
    ):
        return True

    return False


def main() -> None:
    files = collectFiles()
    corpus = buildCorpus(files)
    print(f"Scanned {len(files)} Python files.")

    # ---- Step 1: orphan modules ----
    moduleNames: dict[str, Path] = {}
    for f in files:
        moduleNames[fileToModule(f)] = f

    importedModules: set[str] = set()
    for f in files:
        imps = collectImports(corpus[f], fileToModule(f))
        for imp in imps:
            # Mark every prefix of dotted import as "imported" so that
            # `from src.modules.understanding import x` covers x's package.
            parts = imp.split(".")
            for i in range(1, len(parts) + 1):
                importedModules.add(".".join(parts[:i]))

    orphans: list[dict] = []
    for mod, f in moduleNames.items():
        text = corpus[f]
        if mod in importedModules:
            continue
        if isEntryFile(f, text):
            continue
        orphans.append({
            "module": mod,
            "file": str(f.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "lines": len(text.splitlines()),
        })

    print(f"Orphan modules (never imported, not entry points): {len(orphans)}")

    # ---- Step 2: low-use standalone functions ----
    allDefs: list[dict] = []
    for f in files:
        try:
            tree = ast.parse(corpus[f])
        except SyntaxError:
            continue
        v = DefVisitor(f)
        v.visit(tree)
        allDefs.extend(v.defs)

    # Group by name to flag overloads
    byName: dict[str, int] = defaultdict(int)
    for d in allDefs:
        byName[d["name"]] += 1

    lowUse: list[dict] = []
    for d in allDefs:
        if d["isMethod"] or d["isDunder"]:
            continue
        if isAlwaysUsed(d):
            continue
        if byName[d["name"]] > 1:
            continue  # ambiguous (multiple defs share name)
        defPath = PROJECT_ROOT / d["file"]
        refs = findReferences(d["name"], corpus, defPath)
        # Filter out reference lines that are themselves a `def name(` line
        # in another file (shouldn't happen here since byName>1 was excluded).
        refs = [r for r in refs if not r[2].startswith(f"def {d['name']}(")]
        if len(refs) <= 2:
            d["refs"] = refs
            d["refCount"] = len(refs)
            lowUse.append(d)

    print(f"Low-use standalone functions (<=2 external refs, unique-name): {len(lowUse)}")

    out = {
        "totalFiles": len(files),
        "orphanModules": orphans,
        "lowUseFunctions": lowUse,
    }
    outPath = PROJECT_ROOT / "scripts" / "validation" / "dead_code_audit.json"
    outPath.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {outPath}")

    # ---- Stdout summary ----
    print("\n## Orphan modules\n")
    for o in sorted(orphans, key=lambda x: x["file"]):
        print(f"- `{o['file']}` ({o['lines']} lines)")

    print("\n## Low-use standalone functions\n")
    print("| File:Line | Function | Refs | Sample sites |")
    print("|---|---|---|---|")
    for d in sorted(lowUse, key=lambda x: (x["refCount"], x["file"])):
        sample = "; ".join(f"{r[0]}:{r[1]}" for r in d["refs"][:3]) or "(none)"
        print(f"| {d['file']}:{d['lineno']} | `{d['name']}` | {d['refCount']} | {sample} |")


if __name__ == "__main__":
    main()
