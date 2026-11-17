import ast
from pathlib import Path
from types import MappingProxyType

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src" / "text2motion"
ROOT_PACKAGE = "text2motion"
MAX_FILES = 10
ALLOWED = MappingProxyType(
    {
        "motion": frozenset(),
        "dataset": frozenset({"motion"}),
        "studio": frozenset({"motion"}),
        "tokenization": frozenset({"motion", "dataset"}),
        "generation": frozenset({"motion", "dataset", "tokenization"}),
        "evaluation": frozenset({"motion", "generation"}),
        "app": frozenset(
            {"motion", "dataset", "studio", "tokenization", "generation", "evaluation"}
        ),
    }
)


def package_folders() -> list[Path]:
    return sorted(path for path in SOURCE.iterdir() if path.is_dir() and any(path.rglob("*.py")))


def imported_packages(tree: ast.Module) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == ROOT_PACKAGE:
            modules = [f"{ROOT_PACKAGE}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        for module in modules:
            parts = module.split(".")
            if len(parts) > 1 and parts[0] == ROOT_PACKAGE:
                found.append((node.lineno, parts[1]))
    return found


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def test_each_package_imports_only_the_packages_below_it():
    problems = []
    for folder in package_folders():
        if folder.name not in ALLOWED:
            problems.append(f"package folder {folder.name} is not in ALLOWED")
        allowed = ALLOWED.get(folder.name, frozenset())
        for path in sorted(folder.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for line, package in imported_packages(tree):
                if package != folder.name and package not in allowed:
                    problems.append(f"{relative(path)}:{line}: imports package {package}")
    assert not problems, "\n".join(problems)


def test_each_package_has_at_most_ten_files():
    crowded = []
    for folder in SOURCE.iterdir():
        if not folder.is_dir():
            continue
        count = len(list(folder.glob("*.py")))
        if count > MAX_FILES:
            crowded.append(f"{relative(folder)} has {count} .py files, max is {MAX_FILES}")
    assert not crowded, "\n".join(crowded)
