from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        print(f"WARNING: git commit unavailable ({type(exc).__name__}: {exc}); recording 'unknown'")
        return "unknown"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def start_run(name: str, cfg: Any, outputs_dir: Path, extra: dict | None = None) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(outputs_dir) / "runs" / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch  # core dependency: fail loud if somehow absent, never a silent skip

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
    }

    manifest = {
        "name": name,
        "started_utc": stamp,
        "git_commit": _git_commit(),
        "config": _jsonable(cfg),
        "extra": _jsonable(extra or {}),
        "versions": versions,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def log_metrics(run_dir: Path, record: dict) -> None:
    record = {"logged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **_jsonable(record)}
    with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
