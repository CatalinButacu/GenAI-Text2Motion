import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from text2motion.app.logging_setup import add_run_log_file, ensure_logging_configured

logger = logging.getLogger(__name__)


def __git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.warning(f"git commit unavailable ({type(exc).__name__}: {exc}); recording 'unknown'")
        return "unknown"


def __jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return __jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): __jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [__jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def start_run(
    name: str,
    cfg: Any,
    logs_dir: Path,
    extra: dict | None = None,
    existing_run_dir: Path | None = None,
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    import torch

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
    }

    manifest = {
        "name": name,
        "started_utc": stamp,
        "git_commit": __git_commit(),
        "config": __jsonable(cfg),
        "extra": __jsonable(extra or {}),
        "versions": versions,
    }

    if existing_run_dir is not None:
        run_dir = Path(existing_run_dir)
        if not run_dir.is_dir():
            raise ValueError(f"--resume_run_dir does not exist: {run_dir}")
        resumes_path = run_dir / "resumes.jsonl"
        with resumes_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(manifest) + "\n")
        ensure_logging_configured()
        add_run_log_file(run_dir)
        logger.info(f"run directory {run_dir} (resumed, commit {manifest['git_commit'][:8]})")
        return run_dir

    run_dir = Path(logs_dir) / "train" / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ensure_logging_configured()
    add_run_log_file(run_dir)
    logger.info(f"run directory {run_dir} (commit {manifest['git_commit'][:8]})")
    logger.info(f"versions {versions}")
    return run_dir


def log_metrics(run_dir: Path, record: dict) -> None:
    record = {
        "logged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **__jsonable(record),
    }
    with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
