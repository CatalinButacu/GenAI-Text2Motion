#!/usr/bin/env python
"""PostToolUse hook: format + lint Python files right after Claude edits them.

Reads the tool-call JSON from stdin, extracts the edited file path, and — if it's a
Python file inside this repo — runs `ruff format` then `ruff check --fix`. Unresolved
lint errors are surfaced to Claude via exit code 2 (stderr is shown back to the model).

Wired in .claude/settings.json under hooks.PostToolUse (matcher: Edit|Write).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def edited_path(payload: dict) -> str | None:
    tool_input = payload.get("tool_input") or {}
    return tool_input.get("file_path") or tool_input.get("path")


def main() -> int:
    raw = sys.stdin.read()

    if not raw.strip():
        return 0

    try:
        payload = json.loads(raw)

    except json.JSONDecodeError:
        return 0

    path = edited_path(payload)

    if not path or not path.endswith(".py"):
        return 0

    target = Path(path)

    # Only touch files inside this repo.
    try:
        target.resolve().relative_to(REPO)

    except ValueError:
        return 0

    if not target.exists():
        return 0

    ruff = shutil.which("ruff")

    if ruff is None:
        return 0  # ruff not installed yet (pre-scaffold) — no-op, don't block edits.

    subprocess.run([ruff, "format", str(target)], cwd=REPO, capture_output=True)

    # Lint the whole project so cross-file issues (unused imports, wrong names) are caught.
    src_dirs = [str(REPO / d) for d in ("src", "scripts", "tests") if (REPO / d).exists()]
    check_targets = src_dirs if src_dirs else [str(target)]
    check = subprocess.run(
        [ruff, "check", "--fix"] + check_targets, cwd=REPO, capture_output=True, text=True
    )

    if check.returncode != 0:
        sys.stderr.write("ruff check found unresolved issues:\n")
        sys.stderr.write(check.stdout + check.stderr)
        return 2  # surfaced to Claude as feedback.

    return 0


if __name__ == "__main__":
    sys.exit(main())
