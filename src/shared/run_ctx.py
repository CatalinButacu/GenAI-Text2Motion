"""Training run context: run IDs, GPU sanity, offline wandb, config snapshot."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    WANDB_AVAILABLE = False

log = logging.getLogger(__name__)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def log_gpu_sanity() -> dict[str, Any]:
    """Log + return env details. Written to run log so you can spot T4-instead-of-A100."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["gpu_mem_total_gb"] = round(props.total_memory / 1024**3, 2)

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip()
            info["nvidia_smi"] = out
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

    log.info("[run_ctx] env: %s", json.dumps(info, indent=None))

    return info


def make_run_dir(base: str, run_id: str | None = None) -> tuple[Path, str]:
    """Create checkpoints/<task>/<run_id>/ and return (path, run_id)."""
    run_id = run_id or new_run_id()
    out = Path(base) / run_id
    out.mkdir(parents=True, exist_ok=True)

    return out, run_id


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Best-effort conversion of a config (dataclass instance or mapping) to a plain dict."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)

    return dict(cfg)  # type: ignore[arg-type]


def snapshot_config(run_dir: Path, cfg: Any) -> None:
    """Serialize a dataclass config to run_dir/config.json."""
    payload = config_to_dict(cfg)
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))


def init_wandb(
    project: str, run_id: str, run_dir: Path, config: Any, tags: list[str] | None = None,
):
    """Init wandb. Online when WANDB_API_KEY is set (cloud), offline otherwise.

    Bug fixed 2026-05-22: the previous unconditional
    `setdefault("WANDB_MODE", "offline")` forced offline mode even when
    the cloud startup script had exported a valid WANDB_API_KEY. Online
    runs were silently downgraded to offline.
    """
    # Default to offline ONLY when there's no API key. With a key + no explicit
    # WANDB_MODE override, wandb's own default behavior (online) takes over.
    if "WANDB_API_KEY" not in os.environ:
        os.environ.setdefault("WANDB_MODE", "offline")
    os.environ["WANDB_DIR"] = str(run_dir)
    os.environ["WANDB_SILENT"] = "true"

    if not WANDB_AVAILABLE:
        log.warning("[run_ctx] wandb not installed -- skipping experiment tracking")
        return None

    cfg_dict = config_to_dict(config)
    run = wandb.init(  # type: ignore[union-attr]
        project=project,
        name=run_id,
        id=run_id,
        dir=str(run_dir),
        config=cfg_dict,
        tags=tags or [],
        reinit=True,
    )
    log.info("[run_ctx] wandb offline run initialised: %s (dir=%s)", run_id, run_dir)

    return run


def wandb_log(metrics: dict[str, Any], step: int | None = None) -> None:
    """No-op if wandb is not installed or no active run.

    Wraps wandb.log in a broad except — wandb's offline service uses an inter-
    process socket that Windows occasionally tears down (WinError 64) under
    long-running training. Telemetry is best-effort; never let it kill a run.
    """
    if WANDB_AVAILABLE and wandb.run is not None:  # type: ignore[union-attr]

        try:
            wandb.log(metrics, step=step)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — see docstring
            log.warning("[run_ctx] wandb.log failed (%s): %s", type(exc).__name__, exc)


def wandb_finish() -> None:
    if WANDB_AVAILABLE and wandb.run is not None:  # type: ignore[union-attr]

        try:
            wandb.finish()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log.warning("[run_ctx] wandb.finish failed (%s): %s", type(exc).__name__, exc)


# ---- data integrity ---------------------------------------------------------

def hash_paths(paths: list[Path], chunk: int = 1024 * 1024) -> str:
    """Sha256 over (path, size, first+last chunk) of every file. Fast, not cryptographic."""
    h = hashlib.sha256()

    for p in sorted(paths):
        if not p.is_file():
            continue

        size = p.stat().st_size
        h.update(str(p).encode())
        h.update(size.to_bytes(8, "little"))

        with open(p, "rb") as f:
            h.update(f.read(chunk))

            if size > chunk:
                f.seek(-min(chunk, size), 2)
                h.update(f.read(chunk))

    return h.hexdigest()[:16]


def data_fingerprint(
    roots: list[str], globs: tuple[str, ...] = ("*.npz", "*.pkl", "*.txt")
) -> str:
    """Fingerprint a set of data dirs so training can refuse to start if corpus changed silently."""
    files: list[Path] = []

    for r in roots:
        p = Path(r)

        if not p.exists():
            continue

        for g in globs:
            files.extend(p.rglob(g))

    return hash_paths(files)
