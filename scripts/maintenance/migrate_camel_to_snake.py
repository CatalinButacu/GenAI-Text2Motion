"""One-shot camelCase -> snake_case migration for the project.

What it does:
    1. Walks every .py file under src/, scripts/, tests/, plus main.py.
    2. AST-collects every identifier defined in the project:
       function/method names, parameter names, attribute assignments,
       dataclass field names, top-level variable names.
    3. Filters to identifiers in camelCase (first letter lowercase + at
       least one uppercase). PascalCase class names and ALL_CAPS constants
       are left untouched. dunder names (__foo__) untouched.
    4. Builds a rename map oldName -> old_name using the standard
       insertion-of-underscore rule.
    5. Applies the rename to every .py file via word-boundary regex.
       Yes, this also touches docstring + comment occurrences. That is
       intentional: they should match the new name too.

What it does NOT do:
    - Migrate saved checkpoints. After this runs, old checkpoints saved
      with camelCase state-dict keys will need scripts/refactor/
      migrate_checkpoint_keys.py before they can be loaded.
    - Touch YAML/JSON keys (none of ours use camelCase anyway).
    - Touch external library attribute names (np., torch.*, scipy.*,
      spacy., sentence_transformers., aitviewer.* — all snake_case in
      2025-2026).

Run from repo root:
    python scripts/maintenance/migrate_camel_to_snake.py
    python scripts/maintenance/migrate_camel_to_snake.py --dry-run
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Files / dirs to skip entirely.
SKIP_PARTS = {"__pycache__", ".git", ".venv", ".pytest_cache", "node_modules"}
# Specific files that are themselves migration tooling -- don't rewrite them
# while we're using them.
SKIP_FILES = {
    "scripts/maintenance/migrate_camel_to_snake.py",
    "scripts/refactor/migrate_checkpoint_keys.py",
    "scripts/maintenance/remove_syspath_hacks.py",
}
# Anywhere a camelCase token is part of an EXTERNAL API contract we can't
# touch. Add patterns here if the rename breaks something.
EXTERNAL_NAMES = {
    # torch.amp.autocast / GradScaler arg names
    "device_type",  # already snake -- safe
    # aitviewer attribute names (verified -- all snake_case)
    # spacy attributes
    # No camelCase external names we depend on at present.
}

# Identifiers we observed that conflict with a built-in / external usage
# in a way that would break. Populated empirically during dry-runs.
PROTECTED: set[str] = set()


CAMEL_RE = re.compile(r"^[a-z][a-z0-9]*([A-Z][a-z0-9]*)+$")
ANY_CAMEL_RE = re.compile(r"[a-z][a-z0-9]*([A-Z][a-z0-9]*)+")


def is_camel_case(name: str) -> bool:
    return bool(CAMEL_RE.match(name))


def camel_to_snake(name: str) -> str:
    """maxLength -> max_length, isActor -> is_actor, dModel -> d_model.

    Handles ALL_CAPS-followed-by-lower correctly: XMLParser -> xml_parser.
    """
    # Insert _ between a lowercase/digit and an uppercase
    out = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Insert _ between an uppercase and an uppercase-followed-by-lower
    out = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", out)
    return out.lower()


def should_skip_file(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in SKIP_FILES:
        return True
    return any(part in SKIP_PARTS for part in path.parts)


def collect_defined_names(py_files: list[Path]) -> set[str]:
    """AST-walk to find identifiers defined inside the project."""
    names: set[str] = set()

    for path in py_files:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  skip {path}: {e}")
            continue

        for node in ast.walk(tree):
            # function / method / class definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
                for arg in (*node.args.args, *node.args.kwonlyargs,
                            *node.args.posonlyargs):
                    names.add(arg.arg)
                if node.args.vararg:
                    names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    names.add(node.args.kwarg.arg)
            elif isinstance(node, ast.ClassDef):
                # Class names stay PascalCase; we DON'T add node.name.
                pass
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    collect_assign_target(target, names)
            elif isinstance(node, ast.AnnAssign) and node.target is not None:
                collect_assign_target(node.target, names)
            elif isinstance(node, ast.AugAssign):
                collect_assign_target(node.target, names)
            elif isinstance(node, ast.For):
                collect_assign_target(node.target, names)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                collect_assign_target(node.optional_vars, names)
            elif isinstance(node, ast.NamedExpr):
                # walrus :=
                collect_assign_target(node.target, names)
            elif isinstance(node, ast.comprehension):
                collect_assign_target(node.target, names)
            elif isinstance(node, ast.Lambda):
                for arg in (*node.args.args, *node.args.kwonlyargs):
                    names.add(arg.arg)
            elif isinstance(node, ast.keyword) and node.arg is not None:
                # Keyword argument NAMES at call sites count as identifiers --
                # they must match the function signature, which we're renaming.
                names.add(node.arg)
            elif isinstance(node, ast.Attribute):
                # self.someAttr = ... -- the .someAttr part is identifier-like
                names.add(node.attr)

    return names


def collect_assign_target(target: ast.expr, sink: set[str]) -> None:
    if isinstance(target, ast.Name):
        sink.add(target.id)
    elif isinstance(target, ast.Attribute):
        sink.add(target.attr)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            collect_assign_target(elt, sink)
    elif isinstance(target, ast.Starred):
        collect_assign_target(target.value, sink)


def build_rename_map(names: set[str]) -> dict[str, str]:
    rename: dict[str, str] = {}
    for name in names:
        if name.startswith("__") and name.endswith("__"):
            continue
        if name in EXTERNAL_NAMES or name in PROTECTED:
            continue
        if not is_camel_case(name):
            continue
        snake = camel_to_snake(name)
        if snake == name:
            continue
        rename[name] = snake
    return rename


def apply_rename(path: Path, rename: dict[str, str], dryRun: bool) -> tuple[int, list[str]]:
    """Apply rename map to a single file. Returns (changes, sample hits)."""
    src = path.read_text(encoding="utf-8")
    # Build a single alternation regex of all camelCase keys, word-bounded.
    # Sort longest-first so longer names don't get partially matched.
    keys = sorted(rename.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")

    samples: list[str] = []
    changes = 0

    def sub(match: re.Match) -> str:
        nonlocal changes
        old = match.group(1)
        new = rename[old]
        changes += 1
        if len(samples) < 5:
            samples.append(f"{old} -> {new}")
        return new

    new_src = pattern.sub(sub, src)
    if new_src != src and not dryRun:
        path.write_text(new_src, encoding="utf-8", newline="\n")
    return changes, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan, don't write files.")
    parser.add_argument("--show-map", action="store_true",
                        help="Print the full rename map and exit.")
    args = parser.parse_args()

    py_files = [
        p for p in ROOT.rglob("*.py")
        if not should_skip_file(p)
    ]
    print(f"Found {len(py_files)} .py files in scope.")

    print("Collecting identifiers ...")
    names = collect_defined_names(py_files)
    print(f"  total identifiers defined in project: {len(names)}")

    rename = build_rename_map(names)
    print(f"  camelCase identifiers to rename: {len(rename)}")

    if args.show_map:
        for k in sorted(rename):
            print(f"  {k:<32}  ->  {rename[k]}")
        return

    print("Applying ...")
    total_changes = 0
    touched = 0
    for path in py_files:
        changes, samples = apply_rename(path, rename, dryRun=args.dry_run)
        if changes:
            touched += 1
            total_changes += changes
            rel = path.relative_to(ROOT).as_posix()
            print(f"  {rel:<55}  {changes:>4} renames"
                  + (f"   e.g. {samples[0]}" if samples else ""))

    print()
    print(f"{'DRY-RUN: would touch' if args.dry_run else 'Touched'} "
          f"{touched}/{len(py_files)} files; {total_changes} renames.")

    # Persist the rename map so the user has a record + can use it for
    # checkpoint key migration.
    map_path = ROOT / "scripts" / "maintenance" / "rename_map.json"
    if not args.dry_run:
        map_path.write_text(json.dumps(dict(sorted(rename.items())), indent=2),
                           encoding="utf-8")
        print(f"Wrote rename map -> {map_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
