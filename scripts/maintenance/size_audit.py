# .\.venv\Scripts\python.exe scripts/maintenance/size_audit.py
from __future__ import annotations

import ast
import csv
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from module_freeze import SOURCE_ROOT, frozen_hashes

SOURCES = Path("src")
OUTPUT = Path("docs/notes/size_audit_limits.csv")
KINDS = ("file", "function", "class")
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True, slots=True)
class Limits:
    file_lines: int
    function_lines: int
    class_methods: int

    @property
    def label(self) -> str:
        return f"{self.file_lines}/{self.function_lines}/{self.class_methods}"

    def of_kind(self, kind: str) -> int:
        by_kind = {
            "file": self.file_lines,
            "function": self.function_lines,
            "class": self.class_methods,
        }
        return by_kind[kind]


LIMIT_SETS = (Limits(300, 40, 12), Limits(250, 30, 10))


@dataclass(frozen=True, slots=True)
class Measure:
    kind: str
    package: str
    frozen: str
    path: str
    name: str
    line: int
    size: int


@dataclass(frozen=True, slots=True)
class Offender:
    limits: str
    limit: int
    measure: Measure


def package_of(path: Path) -> str:
    parts = path.relative_to(SOURCE_ROOT).parts

    if len(parts) == 1:
        return SOURCE_ROOT.name

    return parts[0]


def definitions(node: ast.AST, prefix: str) -> Iterator[tuple[str, ast.AST]]:
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (*FUNCTION_NODES, ast.ClassDef)):
            yield from definitions(child, prefix)
            continue

        name = f"{prefix}{child.name}"
        yield name, child
        yield from definitions(child, f"{name}.")


def method_count(node: ast.ClassDef) -> int:
    return sum(isinstance(item, FUNCTION_NODES) for item in node.body)


def file_measures(path: Path, frozen: frozenset[str]) -> Iterator[Measure]:
    text = path.read_text(encoding="utf-8")
    package = package_of(path)
    mark = "yes" if package in frozen else "no"
    location = path.as_posix()
    yield Measure("file", package, mark, location, path.name, 1, len(text.splitlines()))

    for name, node in definitions(ast.parse(text, filename=location), ""):
        if isinstance(node, ast.ClassDef):
            count = method_count(node)
            yield Measure("class", package, mark, location, name, node.lineno, count)
            continue

        if not isinstance(node, FUNCTION_NODES):
            raise TypeError(f"unexpected definition {name} in {location}")

        if node.end_lineno is None:
            raise ValueError(f"no end line for {name} in {location}")

        length = node.end_lineno - node.lineno + 1
        yield Measure("function", package, mark, location, name, node.lineno, length)


def source_measures() -> list[Measure]:
    paths = sorted(SOURCES.rglob("*.py"))

    if not paths:
        raise FileNotFoundError(f"no Python files in {SOURCES}")

    frozen = frozenset(frozen_hashes())

    return [measure for path in paths for measure in file_measures(path, frozen)]


def offenders(measures: list[Measure], limits: Limits) -> list[Offender]:
    found = [
        Offender(limits.label, limits.of_kind(item.kind), item)
        for item in measures
        if item.size > limits.of_kind(item.kind)
    ]

    return sorted(
        found,
        key=lambda row: (KINDS.index(row.measure.kind), -row.measure.size, row.measure.path),
    )


def write_table(rows: list[Offender]) -> None:
    columns = ["limits", "limit", *(field.name for field in fields(Measure))]

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({"limits": row.limits, "limit": row.limit, **asdict(row.measure)})


def print_counts(rows: list[Offender], limits: Limits) -> None:
    chosen = [row.measure for row in rows if row.limits == limits.label]
    counts = Counter((item.package, item.frozen, item.kind) for item in chosen)
    print(f"limits {limits.label}")
    print(f"{'package':<14}{'frozen':<8}{'file':>6}{'function':>10}{'class':>7}")

    for package, frozen in sorted({(item.package, item.frozen) for item in chosen}):
        files, functions, classes = (counts[package, frozen, kind] for kind in KINDS)
        print(f"{package:<14}{frozen:<8}{files:>6}{functions:>10}{classes:>7}")

    files, functions, classes = (sum(item.kind == kind for item in chosen) for kind in KINDS)
    print(f"{'total':<22}{files:>6}{functions:>10}{classes:>7}")


def main() -> None:
    measures = source_measures()
    rows = [row for limits in LIMIT_SETS for row in offenders(measures, limits)]
    write_table(rows)

    for limits in LIMIT_SETS:
        print_counts(rows, limits)

    long_functions = sum(item.kind == "function" and item.size > 80 for item in measures)
    print(f"functions above 80 lines: {long_functions}")
    print(f"wrote {len(rows)} rows to {OUTPUT.as_posix()}")


if __name__ == "__main__":
    main()
