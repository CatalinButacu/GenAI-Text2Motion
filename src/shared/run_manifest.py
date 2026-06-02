from __future__ import annotations

import dataclasses
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("git rev-parse failed: %s", exc)
    return "unavailable"


def build_manifest(config: Any, promptList: list[str], extra: dict | None = None) -> dict:
    """Build a reproducibility manifest for one evaluation or ablation run.

    Parameters
    ----------
    config:
        Any dataclass config object (PipelineConfig or similar).
        Serialised via dataclasses.asdict if it is a dataclass, else str().
    promptList:
        List of prompts used in the run.
    extra:
        Optional additional key-value pairs to include verbatim.

    Returns
    -------
    dict with keys: git_commit, timestamp, python_version, platform,
    config, prompt_count, prompts, and any extra keys.
    """
    try:
        config_dict = dataclasses.asdict(config)
    except TypeError:
        config_dict = str(config)

    manifest: dict = {
        "git_commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config_dict,
        "prompt_count": len(promptList),
        "prompts": promptList,
    }

    if extra:
        manifest.update(extra)

    return manifest


def save_manifest(manifest: dict, output_path: str) -> None:
    """Write manifest dict as indented JSON, creating parent dirs as needed."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
