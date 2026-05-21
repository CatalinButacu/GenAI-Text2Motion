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


def collect_files() -> list[Path]:
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


def file_to_module(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip .py

    return ".".join(parts)


def collect_imports(text: str, file_mod: str) -> set[str]:
    """Return all module-name strings imported by this file (best-effort).
    Resolves relative imports against file_mod (the importing file's dotted name)."""
    imports: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return imports

    parts = file_mod.split(".") if file_mod else []
    pkg_parts = parts[:-1]  # the package this file lives in

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
                # Also record the imported names (for `import a.b.c` add a, a.b, a.b.c)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            base: list[str] = []
            if level > 0:
                # relative — strip `level` parts off pkg_parts
                base = pkg_parts[: max(0, len(pkg_parts) - level + 1)]
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


def is_entry_file(path: Path, text: str) -> bool:
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
        self.defs.append({
            "name": node.name,
            "qualname": qualname,
            "lineno": node.lineno,
            "file": str(self.file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "is_method": bool(self.scopes),
            "decorators": decorators,
            "isPrivate": node.name.startswith("_") and not node.name.startswith("__"),
            "isDunder": node.name.startswith("__") and node.name.endswith("__"),
        })
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()


def build_corpus(files: list[Path]) -> dict[Path, str]:
    return {f: f.read_text(encoding="utf-8", errors="ignore") for f in files}


def find_references(
    name: str, corpus: dict[Path, str], def_file: Path,
) -> list[tuple[str, int, str]]:
    """List every external reference site as (file, line, line_text)."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    out: list[tuple[str, int, str]] = []

    for path, text in corpus.items():
        if path == def_file:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                out.append((rel, i, line.strip()))

    return out


def is_always_used(d: dict) -> bool:
    name = d["name"]
    if d["isDunder"]:
        return True
    if name in {"main", "setUp", "tearDown", "setUpClass", "tearDownClass"}:
        return True
    if name.startswith("test_") or name.startswith("Test"):
        return True
    deco_str = " ".join(d["decorators"])
    if any(tag in deco_str for tag in ("fixture", "parametrize", "given(", "register",
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
    files = collect_files()
    corpus = build_corpus(files)
    print(f"Scanned {len(files)} Python files.")

    # ---- Step 1: orphan modules ----
    module_names: dict[str, Path] = {}
    for f in files:
        module_names[file_to_module(f)] = f

    imported_modules: set[str] = set()
    for f in files:
        imps = collect_imports(corpus[f], file_to_module(f))
        for imp in imps:
            # Mark every prefix of dotted import as "imported" so that
            # `from src.modules.understanding import x` covers x's package.
            parts = imp.split(".")
            for i in range(1, len(parts) + 1):
                imported_modules.add(".".join(parts[:i]))

    orphans: list[dict] = []
    for mod, f in module_names.items():
        text = corpus[f]
        if mod in imported_modules:
            continue
        if is_entry_file(f, text):
            continue
        orphans.append({
            "module": mod,
            "file": str(f.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "lines": len(text.splitlines()),
        })

    print(f"Orphan modules (never imported, not entry points): {len(orphans)}")

    # ---- Step 2: low-use standalone functions ----
    all_defs: list[dict] = []
    for f in files:
        try:
            tree = ast.parse(corpus[f])
        except SyntaxError:
            continue
        v = DefVisitor(f)
        v.visit(tree)
        all_defs.extend(v.defs)

    # Group by name to flag overloads
    by_name: dict[str, int] = defaultdict(int)
    for d in all_defs:
        by_name[d["name"]] += 1

    low_use: list[dict] = []
    for d in all_defs:
        if d["is_method"] or d["isDunder"]:
            continue
        if is_always_used(d):
            continue
        if by_name[d["name"]] > 1:
            continue  # ambiguous (multiple defs share name)
        def_path = PROJECT_ROOT / d["file"]
        refs = find_references(d["name"], corpus, def_path)
        # Filter out reference lines that are themselves a `def name(` line
        # in another file (shouldn't happen here since by_name>1 was excluded).
        refs = [r for r in refs if not r[2].startswith(f"def {d['name']}(")]
        if len(refs) <= 2:
            d["refs"] = refs
            d["refCount"] = len(refs)
            low_use.append(d)

    print(f"Low-use standalone functions (<=2 external refs, unique-name): {len(low_use)}")

    out = {
        "totalFiles": len(files),
        "orphanModules": orphans,
        "lowUseFunctions": low_use,
    }
    out_path = PROJECT_ROOT / "scripts" / "validation" / "dead_code_audit.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # ---- Stdout summary ----
    print("\n## Orphan modules\n")
    for o in sorted(orphans, key=lambda x: x["file"]):
        print(f"- `{o['file']}` ({o['lines']} lines)")

    print("\n## Low-use standalone functions\n")
    print("| File:Line | Function | Refs | Sample sites |")
    print("|---|---|---|---|")
    for d in sorted(low_use, key=lambda x: (x["refCount"], x["file"])):
        sample = "; ".join(f"{r[0]}:{r[1]}" for r in d["refs"][:3]) or "(none)"
        print(f"| {d['file']}:{d['lineno']} | `{d['name']}` | {d['refCount']} | {sample} |")


if __name__ == "__main__":
    main()
