# .\.venv\Scripts\python.exe scripts/maintenance/move_symbols.py docs/project/plans/moves/<step>.json
import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEARCH_DIRS = ("src", "tests", "scripts")
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
KEPT_STATEMENTS = (ast.Import, ast.ImportFrom, ast.Expr)
RATCHET = Path("tests") / "test_size_ratchet.py"
RATCHET_FILE_LIMIT = 250


class MoveError(RuntimeError):
    pass


@dataclass(frozen=True)
class Move:
    source: str
    target: str
    names: tuple[str, ...]
    copies: tuple[str, ...] = ()


@dataclass
class SourceFile:
    path: Path
    newline: str
    lines: list[str]

    @classmethod
    def read(cls, path: Path) -> "SourceFile":
        raw = path.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in raw else "\n"
        return cls(path, newline, raw.replace("\r\n", "\n").split("\n"))

    def text(self) -> str:
        return "\n".join(self.lines)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(self.text().replace("\n", self.newline).encode("utf-8"))


def module_path(root: Path, module: str) -> Path:
    return root / "src" / Path(*module.split(".")).with_suffix(".py")


def defined_names(node: ast.stmt) -> list[str]:
    if isinstance(node, DEFINITIONS):
        return [node.name]

    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]

    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]

    if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
        return [node.name.id]

    return []


def top_level(tree: ast.Module) -> dict[str, ast.stmt]:
    return {name: node for node in tree.body for name in defined_names(node)}


def span(node: ast.stmt, lines: list[str]) -> tuple[int, int]:
    decorators = getattr(node, "decorator_list", [])
    start = min([node.lineno, *(item.lineno for item in decorators)]) - 1

    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1

    return start, node.end_lineno or node.lineno


def used_names(nodes: list[ast.stmt]) -> set[str]:
    return {item.id for node in nodes for item in ast.walk(node) if isinstance(item, ast.Name)}


def import_lines(tree: ast.Module, lines: list[str]) -> list[str]:
    imports = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    return [line for node in imports for line in lines[node.lineno - 1 : node.end_lineno]]


def import_end(tree: ast.Module) -> int:
    ends = [
        node.end_lineno or 0 for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    return max(ends, default=0)


def from_import(module: str, names: list[str]) -> str:
    return f"from {module} import {', '.join(sorted(names))}"


def cut_code(source: SourceFile, nodes: list[ast.stmt]) -> list[str]:
    spans = sorted(span(node, source.lines) for node in nodes)
    code = [line for start, end in spans for line in ["", "", *source.lines[start:end]]]

    for start, end in reversed(spans):
        del source.lines[start:end]

    return code


def add_imports(file: SourceFile, statements: list[str]) -> None:
    if not statements:
        return

    end = import_end(ast.parse(file.text()))
    file.lines[end:end] = statements


def move_code(root: Path, move: Move) -> None:
    source = SourceFile.read(module_path(root, move.source))
    tree = ast.parse(source.text())
    definitions = top_level(tree)
    missing = [name for name in (*move.names, *move.copies) if name not in definitions]

    if missing:
        raise MoveError(f"{move.source} has no top-level {missing}")

    moved = list({id(definitions[name]): definitions[name] for name in move.names}.values())
    moved_names = {name for node in moved for name in defined_names(node)}
    left_behind = set(definitions) - moved_names - set(move.copies)
    needed_from_source = sorted(used_names(moved) & left_behind)
    imports = import_lines(tree, source.lines)
    code = copied_code(root, move, definitions, source.lines) + cut_code(source, moved)
    still_used = sorted(used_names(ast.parse(source.text()).body) & moved_names)
    add_imports(source, [from_import(move.target, still_used)] if still_used else [])
    source.write()
    write_target(root, move, imports, needed_from_source, code)


def copied_code(
    root: Path, move: Move, definitions: dict[str, ast.stmt], lines: list[str]
) -> list[str]:
    path = module_path(root, move.target)
    present = top_level(ast.parse(path.read_text(encoding="utf-8"))) if path.exists() else {}
    spans = sorted(span(definitions[name], lines) for name in move.copies if name not in present)

    return [line for start, end in spans for line in ["", *lines[start:end]]]


def write_target(
    root: Path, move: Move, imports: list[str], needed: list[str], code: list[str]
) -> None:
    path = module_path(root, move.target)
    extra = [from_import(move.source, needed)] if needed else []

    if path.exists():
        target = SourceFile.read(path)
        add_imports(target, [*imports, *extra])
        target.lines.extend(code)
    else:
        ensure_packages(root, path)
        target = SourceFile(path, "\n", [*imports, *extra, *code, ""])

    target.write()


def ensure_packages(root: Path, path: Path) -> None:
    package_root = root / "src" / "text2motion"

    for folder in reversed(path.parents):
        if package_root in folder.parents and not (folder / "__init__.py").exists():
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "__init__.py").write_bytes(b"")


def python_files(root: Path) -> list[Path]:
    return [path for folder in SEARCH_DIRS for path in (root / folder).rglob("*.py")]


def rewrite_import(node: ast.ImportFrom, line: str, move: Move) -> list[str]:
    indent = line[: len(line) - len(line.lstrip())]
    kept = [alias for alias in node.names if alias.name not in move.names]
    moved = [alias for alias in node.names if alias.name in move.names]

    def statement(module: str, aliases: list[ast.alias]) -> str:
        names = ", ".join(
            alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            for alias in aliases
        )
        return f"{indent}from {module} import {names}"

    return [
        statement(module, aliases)
        for module, aliases in ((move.source, kept), (move.target, moved))
        if aliases
    ]


def rewrite_importers(root: Path, move: Move) -> list[Path]:
    touched = []
    patch_target = re.compile(rf"([\"']){re.escape(move.source)}\.({'|'.join(move.names)})\b")

    for path in python_files(root):
        file = SourceFile.read(path)
        tree = ast.parse(file.text())
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == move.source
            and any(alias.name in move.names for alias in node.names)
        ]

        for node in sorted(nodes, key=lambda item: item.lineno, reverse=True):
            start, end = node.lineno - 1, node.end_lineno or node.lineno
            file.lines[start:end] = rewrite_import(node, file.lines[start], move)

        text = patch_target.sub(rf"\g<1>{move.target}.\g<2>", file.text())

        if nodes or text != file.text():
            file.lines = text.split("\n")
            file.write()
            touched.append(path)

    return touched


def module_imports(root: Path, modules: set[str]) -> list[str]:
    found = []

    for path in python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                found += [
                    f"{path.relative_to(root).as_posix()}:{node.lineno} imports module {node.module}.{alias.name}"
                    for alias in node.names
                    if f"{node.module}.{alias.name}" in modules
                ]

            if isinstance(node, ast.Import):
                found += [
                    f"{path.relative_to(root).as_posix()}:{node.lineno} imports module {alias.name}"
                    for alias in node.names
                    if alias.name in modules
                ]

    return found


def delete_if_empty(root: Path, module: str) -> bool:
    path = module_path(root, module)

    if not path.exists():
        return False

    body = ast.parse(path.read_text(encoding="utf-8")).body

    if top_level_code(body):
        return False

    path.unlink()
    return True


def top_level_code(body: list[ast.stmt]) -> bool:
    rest = [node for node in body if not isinstance(node, KEPT_STATEMENTS)]

    if rest and not any(defined_names(node) for node in rest):
        raise MoveError(
            f"unknown top-level statements remain: {[ast.unparse(n)[:60] for n in rest]}"
        )

    return bool(rest)


def check_no_shadowed_module(root: Path, moves: list[Move]) -> None:
    for move in moves:
        path = module_path(root, move.source)

        if path.exists() and path.with_suffix("").is_dir():
            raise MoveError(f"{path} still has code next to package {path.with_suffix('')}")


def relative(module: str) -> str:
    return "/".join(module.split(".")[1:]) + ".py"


def sync_ratchet(root: Path, moves: list[Move], deleted: set[str]) -> None:
    path = root / RATCHET

    if not path.exists():
        return

    file = SourceFile.read(path)
    text = file.text()

    for move in moves:
        for name in move.names:
            text = re.sub(
                rf'"{re.escape(relative(move.source))}::{name}(?=[".])',
                f'"{relative(move.target)}::{name}',
                text,
            )

    for module in deleted:
        text = text.replace(entry(relative(module)), over_limit_entries(root, moves, module))

    file.lines = text.split("\n")
    file.write()


def over_limit_entries(root: Path, moves: list[Move], module: str) -> str:
    targets = sorted({move.target for move in moves if move.source == module})
    long_targets = [
        target
        for target in targets
        if len(module_path(root, target).read_text(encoding="utf-8").splitlines())
        > RATCHET_FILE_LIMIT
    ]

    return "".join(entry(relative(target)) for target in long_targets)


def entry(name: str) -> str:
    return f'            "{name}",\n'


def tidy(root: Path, paths: list[Path], rules: str) -> None:
    existing = sorted({str(path) for path in paths if path.exists()})

    if not existing:
        return

    ruff = [sys.executable, "-m", "ruff"]
    fix = [*ruff, "check", "--fix", "--select", rules, "--quiet", *existing]
    subprocess.run(fix, cwd=root, check=False)
    subprocess.run([*ruff, "format", "--quiet", *existing], cwd=root, check=True)


def load_moves(plan: Path) -> list[Move]:
    entries = json.loads(plan.read_text(encoding="utf-8"))
    return [
        Move(entry["from"], entry["to"], tuple(entry["names"]), tuple(entry.get("copy", ())))
        for entry in entries
    ]


def reexported_names(root: Path, module: str) -> dict[str, list[str]]:
    tree = ast.parse(module_path(root, module).read_text(encoding="utf-8"))
    origins: dict[str, list[str]] = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [alias.name for alias in node.names if alias.asname is None]
            origins.setdefault(node.module, []).extend(names)

    return origins


def apply_moves(root: Path, moves: list[Move]) -> list[str]:
    modules: list[Path] = []
    importers: list[Path] = []
    reexports = {move.source: reexported_names(root, move.source) for move in moves}

    for move in moves:
        move_code(root, move)
        modules += [module_path(root, move.source), module_path(root, move.target)]
        importers += rewrite_importers(root, move)

    deleted = {move.source for move in moves if delete_if_empty(root, move.source)}

    for module in sorted(deleted):
        for origin, names in reexports[module].items():
            importers += rewrite_importers(root, Move(module, origin, tuple(names)))

    check_no_shadowed_module(root, moves)
    tidy(root, modules, "F401,I")
    tidy(root, [path for path in importers if path not in modules], "I")
    sync_ratchet(root, moves, deleted)

    return module_imports(root, deleted)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move top-level symbols and rewrite their imports."
    )
    parser.add_argument("plan", type=Path, help="JSON list of {from, to, names}")
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args()
    warnings = apply_moves(args.root, load_moves(args.plan))

    for warning in warnings:
        print(f"fix by hand: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
