from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from text2motion.motion.model import StreamedChunk
from text2motion.motion.storage import sha256_file

logger = logging.getLogger(__name__)


############################################################
#       MotionStreamProtocol
############################################################


class MotionStreamProtocol:
    @staticmethod
    def send_json(io, obj: dict) -> None:
        io.write((json.dumps(obj) + "\n").encode("utf-8"))
        io.flush()

    @staticmethod
    def recv_json(io) -> dict:
        line = io.readline()
        if not line:
            raise ConnectionError("peer closed")
        return json.loads(line.decode("utf-8"))

    @classmethod
    def send_array(cls, io, kind: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        cls.send_json(io, {"type": kind, "shape": list(arr.shape), "dtype": "float32"})
        io.write(arr.tobytes())
        io.flush()

    @staticmethod
    def recv_array(io, msg: dict) -> np.ndarray:
        count = int(np.prod(msg["shape"]))
        data = io.read(count * 4)
        if not data or len(data) != count * 4:
            raise ConnectionError("short read")
        return np.frombuffer(data, dtype=np.float32).reshape(msg["shape"]).copy()


############################################################
#       MotionInferenceClient
############################################################


class MotionInferenceClient:
    def __init__(self, host: str, port: int, timeout_seconds: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_seconds)
        self.sock.settimeout(None)
        self.io = self.sock.makefile("rwb")
        self.lock = threading.Lock()
        self.hello = MotionStreamProtocol.recv_json(self.io)
        self.last_stats: dict | None = None

    def load(self, config: str, ckpt: str, tokenizer_ckpt: str, backbone: str) -> dict:
        with self.lock:
            MotionStreamProtocol.send_json(
                self.io,
                {
                    "cmd": "load",
                    "config": config,
                    "ckpt": ckpt,
                    "tokenizer_ckpt": tokenizer_ckpt,
                    "backbone": backbone,
                },
            )
            msg = MotionStreamProtocol.recv_json(self.io)
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        self.hello = msg
        return msg

    def generate(
        self,
        prompt: str,
        steps: int,
        temperature: float,
        top_p: float,
        cfg_scale: float,
        should_cancel=None,
    ):
        with self.lock:
            MotionStreamProtocol.send_json(
                self.io,
                {
                    "cmd": "generate",
                    "prompt": prompt,
                    "steps": steps,
                    "temperature": temperature,
                    "top_p": top_p,
                    "cfg_scale": cfg_scale,
                },
            )
            cancel_sent = False
            finished = False
            try:
                while True:
                    msg = MotionStreamProtocol.recv_json(self.io)
                    kind = msg["type"]
                    if kind == "chunk":
                        joints = MotionStreamProtocol.recv_array(self.io, msg)
                        placement = MotionStreamProtocol.recv_json(self.io)
                        features = MotionStreamProtocol.recv_array(
                            self.io, MotionStreamProtocol.recv_json(self.io)
                        )
                        if should_cancel is not None and should_cancel() and not cancel_sent:
                            MotionStreamProtocol.send_json(self.io, {"cmd": "cancel"})
                            cancel_sent = True
                        yield StreamedChunk(
                            joints=joints, features=features, yaw=float(placement["yaw"])
                        )
                    elif kind in ("done", "cancelled"):
                        finished = True
                        self.last_stats = msg.get("stats")
                        return
                    elif kind == "error":
                        finished = True
                        raise RuntimeError(msg["message"])
                    else:
                        finished = True
                        self.close()
                        raise RuntimeError(f"unexpected message {kind!r}; connection closed")
            finally:
                if not finished:
                    self.__abandon_stream(cancel_sent)

    def __abandon_stream(self, cancel_sent: bool) -> None:
        if not cancel_sent:
            MotionStreamProtocol.send_json(self.io, {"cmd": "cancel"})
        while True:
            msg = MotionStreamProtocol.recv_json(self.io)
            kind = msg["type"]
            if kind in ("chunk", "features"):
                MotionStreamProtocol.recv_array(self.io, msg)
            elif kind in ("done", "cancelled", "error"):
                self.last_stats = msg.get("stats")
                return

    def close(self) -> None:
        try:
            self.io.close()
        finally:
            self.sock.close()


def __start_motion_server(
    config: str,
    ckpt: str,
    tokenizer_ckpt: str,
    backbone: str,
    host: str,
    port: int,
    log_path: str | Path,
    service_config: str | Path,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "text2motion.app.cli",
        "serve",
        "--config",
        config,
        "--ckpt",
        ckpt,
        "--tokenizer_ckpt",
        tokenizer_ckpt,
        "--backbone",
        backbone,
        "--host",
        host,
        "--port",
        str(port),
        "--service_config",
        str(service_config),
    ]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "DETACHED_PROCESS", 0)
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH") or "src"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with open(log_path, "ab") as log:
        subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=flags, env=env)
    logger.info(f"[service] spawned; loading the model (log: {log_path})...")


def __await_motion_server(
    host: str, port: int, wait_seconds: int, log_path: str | Path
) -> MotionInferenceClient:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            return MotionInferenceClient(host, port)
        except OSError:
            time.sleep(2)
    raise RuntimeError(
        f"motion service not up on {host}:{port} after {wait_seconds}s; see {log_path}"
    )


def release_identity(config: str, ckpt: str, tokenizer_ckpt: str, backbone: str) -> dict[str, str]:
    return {
        "backbone": str(backbone),
        "config": str(Path(config).resolve()),
        "config_sha256": sha256_file(config),
        "ckpt": str(Path(ckpt).resolve()),
        "ckpt_sha256": sha256_file(ckpt),
        "tokenizer": str(Path(tokenizer_ckpt).resolve()),
        "tokenizer_sha256": sha256_file(tokenizer_ckpt),
    }


def changed_identity_fields(served: object, expected: dict[str, str]) -> list[str]:
    if not isinstance(served, dict):
        return sorted(expected)
    return sorted(key for key in expected if served.get(key) != expected[key])


def motion_service_client(
    config: str,
    ckpt: str,
    tokenizer_ckpt: str,
    backbone: str,
    host: str,
    port: int,
    log_path: str | Path,
    spawn_wait_seconds: int = 240,
    *,
    service_config: str | Path,
    policy: dict[str, object],
) -> MotionInferenceClient:
    try:
        client = MotionInferenceClient(host, port)
        logger.info(f"[service] reusing warm service ({client.hello.get('ckpt')})")
    except OSError:
        __start_motion_server(
            config, ckpt, tokenizer_ckpt, backbone, host, port, log_path, service_config
        )
        client = __await_motion_server(host, port, spawn_wait_seconds, log_path)
    served_policy = client.hello.get("policy")
    if served_policy != policy:
        client.close()
        raise RuntimeError(
            f"the warm service on {host}:{port} runs the policy {served_policy}, but "
            f"{service_config} asks for {policy}; a load cannot change it. "
            "Stop the warm service, then start again."
        )
    expected = release_identity(config, ckpt, tokenizer_ckpt, backbone)
    changed = changed_identity_fields(client.hello.get("identity"), expected)
    if not changed:
        return client
    logger.info(f"[service] switching warm service to {ckpt} (different: {changed})")
    client.load(config, ckpt, tokenizer_ckpt, backbone)
    still_changed = changed_identity_fields(client.hello.get("identity"), expected)
    if still_changed:
        raise RuntimeError(f"the service loaded another release (different: {still_changed})")
    return client
