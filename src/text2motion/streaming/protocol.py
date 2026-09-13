from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from text2motion.motion.model import StreamedChunk


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
            while True:
                msg = MotionStreamProtocol.recv_json(self.io)
                kind = msg["type"]
                if kind == "chunk":
                    joints = MotionStreamProtocol.recv_array(self.io, msg)
                    placement = MotionStreamProtocol.recv_json(self.io)
                    features = MotionStreamProtocol.recv_array(self.io, MotionStreamProtocol.recv_json(self.io))
                    if should_cancel is not None and should_cancel() and not cancel_sent:
                        MotionStreamProtocol.send_json(self.io, {"cmd": "cancel"})
                        cancel_sent = True
                    yield StreamedChunk(
                        joints=joints, features=features, yaw=float(placement["yaw"])
                    )
                elif kind in ("done", "cancelled"):
                    self.last_stats = msg.get("stats")
                    return
                elif kind == "error":
                    raise RuntimeError(msg["message"])
                else:
                    raise RuntimeError(f"unexpected message {kind!r}")

    def close(self) -> None:
        try:
            self.io.close()
        finally:
            self.sock.close()


def connect_or_start_motion_server(
    config: str,
    ckpt: str,
    tokenizer_ckpt: str,
    backbone: str,
    host: str,
    port: int,
    log_path: str | Path,
    spawn_wait_seconds: int = 240,
) -> MotionInferenceClient:
    client = None
    try:
        client = MotionInferenceClient(host, port)
        print(f"[service] reusing warm service ({client.hello.get('ckpt')})", flush=True)
    except OSError:
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
        ]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab")
        flags = getattr(subprocess, "DETACHED_PROCESS", 0)
        env = {key: value for key, value in os.environ.items() if key.startswith("TEXT2MOTION_")}
        for key in ("PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_HOME", "TORCH_HOME"):
            if key in os.environ:
                env[key] = os.environ[key]
        env["PYTHONPATH"] = env.get("PYTHONPATH") or "src"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=flags, env=env)
        print(f"[service] spawned; loading the model (log: {log_path})...", flush=True)
        deadline = time.time() + spawn_wait_seconds
        while time.time() < deadline:
            try:
                client = MotionInferenceClient(host, port)
                break
            except OSError:
                time.sleep(2)
        if client is None:
            raise RuntimeError(
                f"motion service not up on {host}:{port} after {spawn_wait_seconds}s; see {log_path}"
            )
    if Path(str(client.hello.get("ckpt", ""))).resolve() != Path(ckpt).resolve():
        print(f"[service] switching warm service to {ckpt}", flush=True)
        client.load(config, ckpt, tokenizer_ckpt, backbone)
    return client
