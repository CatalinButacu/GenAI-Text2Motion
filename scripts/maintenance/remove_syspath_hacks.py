"""Remove `sys.path.insert(0, ...)` bootstrap hacks across .py files.

After `pip install -e .` (or `uv sync`), the `src` package is importable directly
and these manual sys.path mutations are no longer needed.

Maintenance one-shot. Run from the repo root:
    python scripts/maintenance/remove_syspath_hacks.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Files to skip: ipynb cells (handled separately) and the legacy refactor script.
SKIP = {
    "scripts/cloud/colab/train_colab.ipynb",
    "notebooks/train_m4_colab.ipynb",
    "scripts/refactor/migrate_checkpoint_keys.py",
    # Self-skip.
    "scripts/maintenance/remove_syspath_hacks.py",
}

SYSPATH_RE = re.compile(r"^sys\.path\.(insert|append)\(.*\)\s*\n", re.MULTILINE)


def strip_file(path: Path) -> bool:
    """Return True if the file was modified."""
    rel = path.relative_to(ROOT).as_posix()

    if rel in SKIP:
        return False

    text = path.read_text(encoding="utf-8")

    if "sys.path.insert" not in text and "sys.path.append" not in text:
        return False

    cleaned = SYSPATH_RE.sub("", text)
    # Collapse run-on blank lines left behind by the deletion.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    path.write_text(cleaned, encoding="utf-8")

    return cleaned != text


def main() -> None:
    touched: list[str] = []
    for pyFile in ROOT.rglob("*.py"):
        if any(p in pyFile.parts for p in (".venv", "__pycache__", ".git")):
            continue

        if strip_file(pyFile):
            touched.append(pyFile.relative_to(ROOT).as_posix())

    print(f"Modified {len(touched)} files:")
    for f in touched:
        print(f"  {f}")


if __name__ == "__main__":
    main()
