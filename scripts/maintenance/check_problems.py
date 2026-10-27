# python scripts/maintenance/check_problems.py --all
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
CHECKED_FOLDERS = ("src", "scripts", "tests")
RULES = "E4,E7,E9,F,I,B,SIM,RET,C90,PLR0911,PLR0912,PLW0602,PLW0603"
COMPLEXITY = "lint.mccabe.max-complexity = 15"
READ_ONLY_FLAG = "--check"
CONSTANT_NAME = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
PROTOCOL_NAMES = frozenset({"logger", "pytestmark"})
MUTABLE_NODES = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
MUTABLE_CALLS = frozenset(
    {"list", "dict", "set", "bytearray", "defaultdict", "deque", "OrderedDict", "Counter"}
)


def tool(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(PYTHON), "-m", *arguments], cwd=REPO, capture_output=True, text=True)


def changed_files() -> list[str]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    names = [line[3:].split(" -> ")[-1].strip('"') for line in status.splitlines()]
    return [name for name in names if (REPO / name).is_file()]


def checked(names: list[str]) -> list[str]:
    return [
        name.replace("\\", "/")
        for name in names
        if name.endswith(".py") and Path(name).parts[0] in CHECKED_FOLDERS
    ]


def repaired_by_ruff(files: list[str]) -> None:
    tool("ruff", "format", *files)
    tool("ruff", "check", "--fix", "--select", RULES, "--config", COMPLEXITY, *files)


def lint_problems(files: list[str]) -> list[str]:
    report = tool(
        "ruff",
        "check",
        "--select",
        RULES,
        "--config",
        COMPLEXITY,
        "--output-format",
        "concise",
        *files,
    )
    return [line for line in report.stdout.splitlines() if ".py:" in line]


def type_problems(files: list[str]) -> list[str]:
    report = tool("pyright", "--outputjson", *files)
    diagnostics = json.loads(report.stdout)["generalDiagnostics"]
    problems = []
    for item in diagnostics:
        if item["severity"] != "error":
            continue
        name = Path(item["file"]).resolve().relative_to(REPO).as_posix()
        line = item["range"]["start"]["line"] + 1
        first_line = item["message"].splitlines()[0]
        problems.append(f"{name}:{line}: {first_line}")
    return problems


def python_files(files: list[str]) -> list[Path]:
    found = []
    for name in files:
        path = REPO / name
        if path.is_dir():
            found.extend(sorted(path.rglob("*.py")))
            continue
        found.append(path)
    return found


def is_mutable_value(value: ast.expr) -> bool:
    if isinstance(value, MUTABLE_NODES):
        return True
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return False
    return value.func.id in MUTABLE_CALLS


def module_assignments(tree: ast.Module) -> list[tuple[int, str, ast.expr]]:
    assignments = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        assignments.extend((node.lineno, name, node.value) for name in names)
    return assignments


def module_name_problem(name: str, value: ast.expr) -> str | None:
    if name.startswith("__") and name.endswith("__"):
        return None
    if name in PROTOCOL_NAMES:
        return None
    if not CONSTANT_NAME.match(name):
        return f"G02 module name {name} is not an UPPER_CASE constant"
    if not is_mutable_value(value):
        return None
    return (
        f"G01 module constant {name} holds a mutable value; "
        "use a tuple, frozenset or MappingProxyType"
    )


def module_name_problems(tree: ast.Module) -> list[tuple[int, str]]:
    problems = []
    for line, name, value in module_assignments(tree):
        problem = module_name_problem(name, value)
        if problem is None:
            continue
        problems.append((line, problem))
    return problems


def state_problems(files: list[str]) -> list[str]:
    problems = []
    for path in python_files(files):
        tree = ast.parse(path.read_bytes().decode("utf-8"), filename=str(path))
        name = path.resolve().relative_to(REPO).as_posix()
        found = module_name_problems(tree)
        problems.extend(f"{name}:{line}: {message}" for line, message in sorted(found))
    return problems


def selected_files(arguments: list[str]) -> list[str]:
    if arguments == ["--all"]:
        return [str(folder) for folder in CHECKED_FOLDERS]
    if arguments:
        return checked(arguments)
    return checked(changed_files())


def main() -> int:
    arguments = sys.argv[1:]
    read_only = READ_ONLY_FLAG in arguments
    files = selected_files([argument for argument in arguments if argument != READ_ONLY_FLAG])
    if not files:
        print("no Python file to check")
        return 0
    if not read_only:
        repaired_by_ruff(files)
    problems = lint_problems(files) + type_problems(files) + state_problems(files)
    for problem in problems:
        print(problem)
    stage = "with no repair" if read_only else "after the automatic repair"
    print(f"{len(problems)} problem(s) remain {stage}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
