from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from text2motion.app.service.protocol import Release, ServiceAddress


def motion_server_command(
    release: Release,
    address: ServiceAddress,
    service_config: str | Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "text2motion.app.cli",
        "serve",
        "--config",
        release.config,
        "--ckpt",
        release.ckpt,
        "--tokenizer_ckpt",
        release.tokenizer_ckpt,
        "--backbone",
        release.backbone,
        "--host",
        address.host,
        "--port",
        str(address.port),
        "--service_config",
        str(service_config),
    ]


def motion_server_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH") or "src"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def spawn_detached_process(cmd: list[str], env: dict[str, str], log_path: str | Path) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "DETACHED_PROCESS", 0)
    with open(log_path, "ab") as log:
        subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=flags, env=env)
