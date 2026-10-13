# .\.venv\Scripts\python.exe scripts/maintenance/rename_private_functions.py --apply
from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_FOLDERS = ("src", "scripts", "tests")
FunctionNode = (ast.FunctionDef, ast.AsyncFunctionDef)
CHOSEN_PUBLIC_NAMES = {
    "_fit_chunk": "fit_single_chunk",
    "_model_root": "resolve_model_root",
    "_tensor": "to_float_tensor",
}


@dataclass
class NameUsage:
    module_definitions: int = 0
    method_owners: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)


def is_single_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__") and len(name) > 1


def python_files() -> list[Path]:
    return sorted(path for folder in SOURCE_FOLDERS for path in Path(folder).rglob("*.py"))


def collect_definitions(trees: dict[Path, ast.Module]) -> dict[str, NameUsage]:
    usages: dict[str, NameUsage] = defaultdict(NameUsage)
    for tree in trees.values():
        class_bodies = {
            id(item): node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for item in node.body
        }
        for node in ast.walk(tree):
            if not isinstance(node, FunctionNode) or not is_single_private(node.name):
                continue
            owner = class_bodies.get(id(node))
            if owner is None:
                usages[node.name].module_definitions += 1
            else:
                usages[node.name].method_owners.add(owner)
    return usages


def class_ancestors(trees: dict[Path, ast.Module]) -> dict[str, set[str]]:
    direct: dict[str, set[str]] = defaultdict(set)
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        direct[node.name].add(base.id)
                    if isinstance(base, ast.Attribute):
                        direct[node.name].add(base.attr)
    resolved: dict[str, set[str]] = {}
    for name in direct:
        pending = list(direct[name])
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(direct.get(current, ()))
        resolved[name] = seen
    return resolved


def find_hazards(trees: dict[Path, ast.Module], usages: dict[str, NameUsage]) -> None:
    ancestors = class_ancestors(trees)
    for name, usage in usages.items():
        owners = sorted(usage.method_owners)
        for first in owners:
            for second in owners:
                if first != second and second in ancestors.get(first, set()):
                    usage.reasons.add(f"overridden: {first} extends {second}")
        if usage.module_definitions and usage.method_owners:
            usage.reasons.add("both a module function and a method")
    for path, tree in trees.items():
        source = path.read_text(encoding="utf-8")
        for name in usages:
            if re.search(rf"[\"']{re.escape(name)}[\"']", source):
                usages[name].reasons.add(f"used as a string in {path.name}")
        class_spans = [
            (
                node.lineno,
                node.end_lineno or node.lineno,
                node.name,
                {item.name for item in node.body if isinstance(item, FunctionNode)},
            )
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in usages:
                enclosing = [span for span in class_spans if span[0] <= node.lineno <= span[1]]
                if enclosing:
                    usages[node.id].reasons.add(f"module name used inside class {enclosing[-1][2]}")
            if isinstance(node, ast.Attribute) and node.attr in usages:
                enclosing = [span for span in class_spans if span[0] <= node.lineno <= span[1]]
                innermost = max(enclosing, key=lambda span: span[0]) if enclosing else None
                receiver_is_own = isinstance(node.value, ast.Name) and node.value.id in (
                    "self",
                    "cls",
                )
                if innermost is None or node.attr not in innermost[3] or not receiver_is_own:
                    usages[node.attr].reasons.add(f"reached from outside its class in {path.name}")
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in usages and any(
                        span[0] <= node.lineno <= span[1] for span in class_spans
                    ):
                        usages[alias.name].reasons.add("imported inside a class")


def all_identifiers(trees: dict[Path, ast.Module]) -> set[str]:
    identifiers: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            if isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            if isinstance(node, (*FunctionNode, ast.ClassDef)):
                identifiers.add(node.name)
            if isinstance(node, ast.arg):
                identifiers.add(node.arg)
            if isinstance(node, ast.alias):
                identifiers.add(node.asname or node.name)
            if isinstance(node, ast.keyword) and node.arg:
                identifiers.add(node.arg)
    return identifiers


def build_plan(usages: dict[str, NameUsage], identifiers: set[str]) -> tuple[dict[str, str], dict]:
    renames: dict[str, str] = {}
    conflicts: dict[str, str] = {}
    for name, usage in sorted(usages.items()):
        target = name.lstrip("_") if usage.reasons else f"_{name}"
        target = CHOSEN_PUBLIC_NAMES.get(name, target)
        if target in identifiers:
            conflicts[name] = target
            continue
        renames[name] = target
    return renames, conflicts


def rewrite(path: Path, renames: dict[str, str]) -> int:
    source = path.read_text(encoding="utf-8")
    replaced = 0
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in renames:
            token = token._replace(string=renames[token.string])
            replaced += 1
        tokens.append(token)
    if not replaced:
        return 0
    lines = source.splitlines(keepends=True)
    for token in reversed(tokens):
        if token.type != tokenize.NAME:
            continue
        row, column = token.start
        original_end = token.end[1]
        line = lines[row - 1]
        if line[column:original_end] != token.string:
            lines[row - 1] = line[:column] + token.string + line[original_end:]
    path.write_text("".join(lines), encoding="utf-8", newline="")
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in python_files()}
    usages = collect_definitions(trees)
    find_hazards(trees, usages)
    renames, conflicts = build_plan(usages, all_identifiers(trees))

    to_double = {old: new for old, new in renames.items() if new.startswith("__")}
    to_public = {old: new for old, new in renames.items() if not new.startswith("__")}
    print(f"private functions: {len(usages)}")
    print(f"  -> double underscore: {len(to_double)}")
    print(f"  -> public (a double underscore would break at run time): {len(to_public)}")
    for old, new in to_public.items():
        print(f"       {old} -> {new}   [{'; '.join(sorted(usages[old].reasons))[:110]}]")
    print(f"  -> left unchanged, the new name already exists: {len(conflicts)}")
    for old, target in conflicts.items():
        print(f"       {old} (wanted {target})   [{'; '.join(sorted(usages[old].reasons))[:90]}]")

    if not args.apply:
        print("dry run: pass --apply to rename")
        return
    total = sum(rewrite(path, renames) for path in trees)
    print(f"renamed {total} identifiers in {len(trees)} files")


if __name__ == "__main__":
    main()
