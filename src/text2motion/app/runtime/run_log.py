import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def git_head_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.warning(f"git commit unavailable ({type(exc).__name__}: {exc}); recording 'unknown'")
        return "unknown"


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


MLFLOW_RUN_ID_FILE = "mlflow_run_id"


def mlflow_module():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow_tracking is on, but MLflow is not installed: run uv sync --extra tracking"
        ) from exc
    return mlflow


def start_mlflow_run(run_dir: Path, logs_dir: Path, manifest: dict) -> None:
    mlflow = mlflow_module()
    mlflow.set_tracking_uri(f"sqlite:///{(Path(logs_dir) / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment(manifest["name"])
    id_file = run_dir / MLFLOW_RUN_ID_FILE
    earlier_id = id_file.read_text(encoding="utf-8") if id_file.exists() else None
    run = mlflow.start_run(run_id=earlier_id, run_name=run_dir.name)
    id_file.write_text(run.info.run_id, encoding="utf-8")
    mlflow.log_params({key: str(value)[:500] for key, value in manifest["extra"].items()})


def log_mlflow_metrics(record: dict) -> None:
    mlflow = mlflow_module()
    numbers = {
        key: float(value)
        for key, value in record.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    mlflow.log_metrics(numbers, step=int(record.get("step", record.get("epoch", 0))))


def resumed_run_dir(resume_path: Path, requested: Path | None) -> Path | None:
    if not resume_path.is_file():
        raise FileNotFoundError(
            f"--resume continues from {resume_path}, which does not exist; start without --resume"
        )

    import torch

    state = torch.load(resume_path, map_location="cpu", weights_only=True, mmap=True)
    recorded = None if state.get("run_dir") is None else Path(state["run_dir"])
    if recorded is not None and requested is not None and recorded.resolve() != requested.resolve():
        raise ValueError(
            f"--resume_run_dir {requested} differs from the run directory {recorded} "
            f"that {resume_path} records"
        )
    if recorded is None and requested is None:
        logger.warning(
            f"{resume_path} does not record its run directory (old resume file); "
            "this resume starts a new run directory"
        )
    return recorded or requested


def run_dir_to_continue(
    resume: bool, fresh_selection: bool, resume_path: Path, requested: Path | None
) -> Path | None:
    if not resume or fresh_selection:
        return requested
    return resumed_run_dir(resume_path, requested)


def run_versions() -> dict:
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
    }


def build_manifest(name: str, stamp: str, cfg: Any, extra: dict | None) -> dict:
    versions = run_versions()
    return {
        "name": name,
        "started_utc": stamp,
        "git_commit": git_head_hash(),
        "config": jsonable(cfg),
        "extra": jsonable(extra or {}),
        "versions": versions,
    }


def checked_existing_run_dir(existing_run_dir: Path) -> Path:
    run_dir = Path(existing_run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"--resume_run_dir does not exist: {run_dir}")
    return run_dir


def append_resume_manifest(run_dir: Path, manifest: dict) -> None:
    resumes_path = run_dir / "resumes.jsonl"
    with resumes_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest) + "\n")


def attach_run_outputs(run_dir: Path, cfg: Any, logs_dir: Path, manifest: dict) -> None:
    ensure_logging_configured()
    add_run_log_file(run_dir)
    if getattr(cfg, "mlflow_tracking", False):
        start_mlflow_run(run_dir, logs_dir, manifest)


def continue_run(existing_run_dir: Path, cfg: Any, logs_dir: Path, manifest: dict) -> Path:
    run_dir = checked_existing_run_dir(existing_run_dir)
    append_resume_manifest(run_dir, manifest)
    attach_run_outputs(run_dir, cfg, logs_dir, manifest)
    logger.info(f"run directory {run_dir} (resumed, commit {manifest['git_commit'][:8]})")
    return run_dir


def create_run(name: str, stamp: str, cfg: Any, logs_dir: Path, manifest: dict) -> Path:
    run_dir = Path(logs_dir) / "train" / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    attach_run_outputs(run_dir, cfg, logs_dir, manifest)
    logger.info(f"run directory {run_dir} (commit {manifest['git_commit'][:8]})")
    logger.info(f"versions {manifest['versions']}")
    return run_dir


def start_run(
    name: str,
    cfg: Any,
    logs_dir: Path,
    extra: dict | None = None,
    existing_run_dir: Path | None = None,
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    manifest = build_manifest(name, stamp, cfg, extra)
    if existing_run_dir is not None:
        return continue_run(existing_run_dir, cfg, logs_dir, manifest)
    return create_run(name, stamp, cfg, logs_dir, manifest)


def log_metrics(run_dir: Path, record: dict) -> None:
    record = {
        "logged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **jsonable(record),
    }
    with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    if (Path(run_dir) / MLFLOW_RUN_ID_FILE).exists():
        log_mlflow_metrics(record)


LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


RUN_LOG_NAME = "run.log"


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configure_logging(level: str = "INFO") -> None:
    if level not in LOG_LEVELS:
        raise ValueError(f"unknown log level {level!r} (expected one of {LOG_LEVELS})")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(console)
    root.setLevel(level)


def ensure_logging_configured() -> None:
    if not logging.getLogger().handlers:
        configure_logging()


def add_run_log_file(run_dir: Path) -> None:
    run_log = logging.FileHandler(Path(run_dir) / RUN_LOG_NAME, encoding="utf-8")
    run_log.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logging.getLogger().addHandler(run_log)
