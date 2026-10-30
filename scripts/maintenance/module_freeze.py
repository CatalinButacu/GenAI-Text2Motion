# .\.venv\Scripts\python.exe scripts/maintenance/module_freeze.py motion
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

SOURCE_ROOT = Path("src/text2motion")
LEDGER = Path("doc/notes/module_ledger.csv")
FREEZE_KINDS = frozenset({"freeze", "unfreeze"})


def module_hash(module: str) -> str:
    folder = SOURCE_ROOT / module
    files = sorted(folder.glob("*.py"))
    if not files:
        raise FileNotFoundError(f"no Python files in {folder}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    latest: dict[str, tuple[str, str]] = {}
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] in FREEZE_KINDS:
                latest[row["module"]] = (row["kind"], row["item"])
    return {module: item for module, (kind, item) in latest.items() if kind == "freeze"}


def main() -> None:
    parser = argparse.ArgumentParser(description="print the freeze hash of one package")
    parser.add_argument("module")
    print(module_hash(parser.parse_args().module))


if __name__ == "__main__":
    main()
