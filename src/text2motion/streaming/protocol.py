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


class Wire:
    HOST: str = "127.0.0.1"
    PORT: int = 8765

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


class MotionServiceClient:
    def __init__(self, host: str = Wire.HOST, port: int = Wire.PORT):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(None)
        self.io = self.sock.makefile("rwb")
        self.lock = threading.Lock()
        self.hello = Wire.recv_json(self.io)

    def load(self, config: str, ckpt: str, tokenizer_ckpt: str, backbone: str) -> dict:
        with self.lock:
            Wire.send_json(
                self.io,
                {
                    "cmd": "load",
                    "config": config,
                    "ckpt": ckpt,
                    "tokenizer_ckpt": tokenizer_ckpt,
                    "backbone": backbone,
                },
            )
            msg = Wire.recv_json(self.io)
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
            Wire.send_json(
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
                msg = Wire.recv_json(self.io)
                kind = msg["type"]
                if kind == "chunk":
                    joints = Wire.recv_array(self.io, msg)
                    if should_cancel is not None and should_cancel() and not cancel_sent:
                        Wire.send_json(self.io, {"cmd": "cancel"})
                        cancel_sent = True
                    yield joints
                elif kind in ("done", "cancelled"):
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


def connect_or_spawn(
    config: str,
    ckpt: str,
    tokenizer_ckpt: str,
    backbone: str,
    host: str = Wire.HOST,
    port: int = Wire.PORT,
    log_path: str = "outputs/motion_service.log",
    spawn_wait_seconds: int = 240,
) -> MotionServiceClient:
    client = None
    try:
        client = MotionServiceClient(host, port)
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
        env = dict(os.environ)
        env["PYTHONPATH"] = env.get("PYTHONPATH") or "src"
        subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=flags, env=env)
        print(f"[service] spawned; loading the model (log: {log_path})...", flush=True)
        deadline = time.time() + spawn_wait_seconds
        while time.time() < deadline:
            try:
                client = MotionServiceClient(host, port)
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
