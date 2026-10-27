# python scripts/maintenance/rename_everywhere.py OLD_NAME NEW_NAME [--only FILE ...]
import argparse
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKIPPED_FOLDERS = {
    ".git",
    ".venv",
    "third_party",
    "artifacts",
    "data",
    "graphify-out",
    "build",
    "__pycache__",
    "text2motion.egg-info",
}
KEPT_FILES = {
    ".claude/docs/STATUS.md",
    ".claude/docs/replay.log",
    "doc/notes/research-log.md",
    "tests/app/test_rename_everywhere.py",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".tex",
    ".yaml",
    ".yml",
    ".toml",
    ".ps1",
    ".sh",
    ".dot",
    ".txt",
    ".bib",
    "",
}
IMPORTED_MODULE = re.compile(r"^\s*(?:from|import) (text2motion(?:\.\w+)+)", re.M)


def replacements(old: str, new: str) -> list[tuple[re.Pattern, str]]:
    if not old.endswith(".py"):
        return [(re.compile(rf"\b{re.escape(old)}\b"), new)]
    old_parts = Path(old).with_suffix("").parts[-2:]
    new_parts = Path(new).with_suffix("").parts[-2:]
    return [
        (re.compile(rf"\b{old_parts[0]}\.{old_parts[1]}\b(?!\.py)"), ".".join(new_parts)),
        (re.compile(rf"\b{old_parts[0]}/{old_parts[1]}\.py\b"), "/".join(new_parts) + ".py"),
        (re.compile(rf"\b{old_parts[0]}\\{old_parts[1]}\.py\b"), "\\\\".join(new_parts) + ".py"),
    ]


def text_files(root: Path) -> list[Path]:
    found = []
    for folder, subfolders, names in os.walk(root):
        subfolders[:] = [name for name in subfolders if name not in SKIPPED_FOLDERS]
        for name in names:
            path = Path(folder) / name
            if path.suffix in TEXT_SUFFIXES and name != "uv.lock":
                found.append(path)
    return found


def limited_files(root: Path, only: list[str]) -> list[Path]:
    paths = [root / name for name in only]
    missing = [name for name, path in zip(only, paths, strict=True) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"--only names files that do not exist: {missing}")
    return paths


def rename_in_text(
    root: Path, old: str, new: str, only: list[str] | None = None
) -> tuple[list[str], list[str]]:
    patterns = replacements(old, new)
    changed = []
    kept = []
    candidates = text_files(root) if only is None else limited_files(root, only)
    for path in candidates:
        try:
            original = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for pattern, replacement in patterns:
            updated = pattern.sub(replacement, updated)
        if updated == original:
            continue
        name = path.relative_to(root).as_posix()
        if name in KEPT_FILES:
            kept.append(name)
            continue
        path.write_bytes(updated.encode("utf-8"))
        changed.append(name)
    return changed, kept


def missing_modules(root: Path) -> list[str]:
    missing = []
    for path in text_files(root):
        if path.suffix != ".py":
            continue
        for module in IMPORTED_MODULE.findall(path.read_text(encoding="utf-8", errors="replace")):
            location = root / "src" / Path(*module.split("."))
            if not location.is_dir() and not location.with_suffix(".py").is_file():
                missing.append(f"{path.relative_to(root).as_posix()}: {module}")
    return missing


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename a symbol or a file in code and documents.")
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument(
        "--only",
        nargs="+",
        help="change only these files, for a name that also has an unrelated meaning elsewhere",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    old, new = args.old, args.new
    if old.endswith(".py") and (REPO / old).is_file():
        (REPO / old).rename(REPO / new)
    changed, kept = rename_in_text(REPO, old, new, args.only)
    print(f"{old} -> {new}: {len(changed)} files updated")
    for name in changed:
        print(f"  {name}")
    for name in kept:
        print(f"  not changed (history record or test data): {name}")
    missing = missing_modules(REPO)
    for line in missing:
        print(f"MISSING MODULE {line}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
