# python scripts/maintenance/check_problems.py --all
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
CHECKED_FOLDERS = ("src", "scripts", "tests")
RULES = "E4,E7,E9,F,I,B,SIM,RET,C90,PLR0911,PLR0912"
COMPLEXITY = "lint.mccabe.max-complexity = 15"
READ_ONLY_FLAG = "--check"


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
    problems = lint_problems(files) + type_problems(files)
    for problem in problems:
        print(problem)
    stage = "with no repair" if read_only else "after the automatic repair"
    print(f"{len(problems)} problem(s) remain {stage}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
