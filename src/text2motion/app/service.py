from __future__ import annotations

import gc
import json
import logging
import os
import select
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from text2motion.generation.decoder import StreamingMotionDecoder
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.contracts import StreamedChunk
from text2motion.motion.representation import FPS, StreamingSkeletonRecovery
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


def process_memory_gb() -> tuple[float, float]:
    import psutil

    process = psutil.Process()
    memory = process.memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    return memory.rss / 1e9, peak / 1e9


def record_inference(row: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        print(json.dumps(row), file=handle)


def format_stats(stats: dict) -> str:
    return (
        f"{stats['frames']} frames / {stats['steps']} steps in {stats['total_ms'] / 1e3:.2f}s "
        f"({stats['realtime_factor']:.1f}x realtime) | "
        f"encode {stats['text_encode_ms']:.0f}ms "
        f"generate {stats['generate_ms']:.0f}ms ({stats['ms_per_step']:.1f} ms/step) "
        f"detokenize {stats['detokenize_ms']:.0f}ms "
        f"send+client-wait {stats['send_ms']:.0f}ms | "
        f"peak gpu {stats['peak_gpu_gb']:.2f} GB  host rss {stats['host_rss_gb']:.2f} GB "
        f"(peak {stats['host_peak_gb']:.2f} GB)"
    )


############################################################
#       LoadedInferencePipeline
############################################################
@dataclass(frozen=True)
class LoadedInferencePipeline:
    generator: TextToMotionGenerator
    decoder: StreamingMotionDecoder
    backbone: str
    checkpoint: str
    downsample: int
    trained_steps: int
    max_steps: int
    device: str
    identity: dict[str, str]
    anchor_every: int = 0
    anchor_context: int = 16

    def hello(self) -> dict:
        return {
            "type": "hello",
            "identity": self.identity,
            "ckpt": self.checkpoint,
            "backbone": self.backbone,
            "downsample": self.downsample,
            "trained_steps": self.trained_steps,
            "max_steps": self.max_steps,
            "device": self.device,
        }


type InferencePipelineLoader = Callable[[dict], LoadedInferencePipeline]


def pending_request(sock: socket.socket, io) -> dict | None:
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return None
    return MotionStreamProtocol.recv_json(io)


############################################################
#       MotionInferenceServer
############################################################
class MotionInferenceServer:
    def __init__(
        self,
        pipeline: LoadedInferencePipeline,
        pipeline_loader: InferencePipelineLoader | None = None,
        inference_log: str | Path | None = None,
        *,
        policy: dict[str, object],
    ) -> None:
        self.pipeline = pipeline
        self.policy = policy
        self.pipeline_loader = pipeline_loader
        self.inference_log = Path(inference_log) if inference_log else None
        self.deferred_request: dict | None = None

    def handle_generate(self, sock: socket.socket, io, request: dict) -> None:
        if self.pipeline is None:
            MotionStreamProtocol.send_json(
                io,
                {
                    "type": "error",
                    "message": "no model loaded -- the last load failed and freed the previous "
                    "one; pick a model and press Load model again",
                },
            )
            return
        frames = 0
        cancelled = False
        started = time.perf_counter()

        recovery = StreamingSkeletonRecovery()
        on_cuda = self.pipeline.device == "cuda"
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        timing = {"generate": 0.0, "send": 0.0, "tokens": 0}

        def synchronize() -> None:
            if on_cuda:
                torch.cuda.synchronize()

        def timed_tokens(token_source):
            while True:
                token_started = time.perf_counter()
                tokens = next(token_source, None)
                synchronize()
                timing["generate"] += time.perf_counter() - token_started
                if tokens is None:
                    return
                timing["tokens"] += 1
                yield tokens

        encode_started = time.perf_counter()
        with torch.no_grad():
            text_emb = self.pipeline.generator.text_encoder([request["prompt"]]).to(
                self.pipeline.device
            )
            synchronize()
            encode_seconds = time.perf_counter() - encode_started
            token_source = self.pipeline.generator.token_generator.stream_token_indices(
                text_emb,
                int(request["steps"]),
                temperature=float(request["temperature"]),
                top_p=float(request["top_p"]),
                cfg_scale=float(request["cfg_scale"]),
                stop_at_end=False,
                anchor_every=self.pipeline.anchor_every,
                anchor_context=self.pipeline.anchor_context,
            )
            for chunk in self.pipeline.decoder.stream_tokens(timed_tokens(token_source)):
                features = chunk.squeeze(0).cpu().numpy()
                carried_yaw = recovery.yaw
                joints = recovery(features)
                frames += len(joints)
                send_started = time.perf_counter()
                MotionStreamProtocol.send_array(io, "chunk", joints)
                MotionStreamProtocol.send_json(io, {"type": "placement", "yaw": carried_yaw})
                MotionStreamProtocol.send_array(io, "features", features)
                timing.setdefault("first_chunk", time.perf_counter() - started)
                interrupting = pending_request(sock, io)
                timing["send"] += time.perf_counter() - send_started
                cancelled = interrupting is not None
                if interrupting is not None and interrupting.get("cmd") != "cancel":
                    logger.info(
                        f"[service] {interrupting.get('cmd')!r} arrived mid-stream -> stream cancelled, request deferred"
                    )
                    self.deferred_request = interrupting
                if cancelled:
                    break

        elapsed = time.perf_counter() - started
        logger.info(
            f"[service] {'cancelled' if cancelled else 'done'} {frames} frames in {elapsed:.2f}s "
            f"({request['prompt']!r})"
        )
        stats = self.inference_stats(elapsed, encode_seconds, timing, frames, on_cuda)
        logger.info(f"[perf] {format_stats(stats)}")
        if self.inference_log is not None:
            record_inference(
                {
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "prompt": request["prompt"],
                    "backbone": self.pipeline.backbone,
                    "checkpoint": self.pipeline.checkpoint,
                    "identity": self.pipeline.identity,
                    "policy": self.policy,
                    "device": self.pipeline.device,
                    "cancelled": cancelled,
                    "requested_steps": int(request["steps"]),
                    "cfg_scale": float(request["cfg_scale"]),
                    **stats,
                },
                self.inference_log,
            )
        if on_cuda:
            torch.cuda.empty_cache()
        MotionStreamProtocol.send_json(
            io,
            {
                "type": "cancelled" if cancelled else "done",
                "frames": frames,
                "stats": stats,
            },
        )

    def inference_stats(
        self, elapsed: float, encode: float, timing: dict, frames: int, on_cuda: bool
    ) -> dict:
        steps = max(1, timing["tokens"])
        host_rss, host_peak = process_memory_gb()
        generate = timing["generate"]
        send = timing["send"]
        detokenize = max(0.0, elapsed - encode - generate - send)
        return {
            "frames": frames,
            "steps": steps,
            "total_ms": elapsed * 1e3,
            "first_chunk_ms": timing.get("first_chunk", elapsed) * 1e3,
            "text_encode_ms": encode * 1e3,
            "generate_ms": generate * 1e3,
            "detokenize_ms": detokenize * 1e3,
            "send_ms": send * 1e3,
            "ms_per_step": generate / steps * 1e3,
            "realtime_factor": (frames / FPS) / elapsed if elapsed > 0 else 0.0,
            "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if on_cuda else 0.0,
            "host_rss_gb": host_rss,
            "host_peak_gb": host_peak,
        }

    def handle_load(self, io, request: dict) -> None:
        if self.pipeline_loader is None:
            MotionStreamProtocol.send_json(
                io, {"type": "error", "message": "this service cannot reload models"}
            )
            return
        previous = self.pipeline
        self.pipeline = None
        del previous
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.pipeline = self.pipeline_loader(request)
        if self.pipeline.device == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"[service] loaded {self.pipeline.checkpoint} ({self.pipeline.backbone})")
        MotionStreamProtocol.send_json(
            io, {**self.pipeline.hello(), "policy": self.policy, "type": "loaded"}
        )

    def serve_client(self, conn: socket.socket, io) -> bool:
        hello = (
            self.pipeline.hello()
            if self.pipeline is not None
            else {"type": "hello", "ckpt": "", "backbone": None}
        )
        MotionStreamProtocol.send_json(io, {**hello, "policy": self.policy})
        self.deferred_request = None
        while True:
            request = self.deferred_request or MotionStreamProtocol.recv_json(io)
            self.deferred_request = None
            command = request.get("cmd")

            if command == "generate":
                try:
                    self.handle_generate(conn, io, request)
                except (ConnectionError, BrokenPipeError):
                    raise
                except Exception as exc:
                    traceback.print_exc()
                    MotionStreamProtocol.send_json(
                        io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                    )
            elif command == "load":
                try:
                    self.handle_load(io, request)
                except Exception as exc:
                    traceback.print_exc()
                    MotionStreamProtocol.send_json(
                        io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                    )
            elif command == "shutdown":
                MotionStreamProtocol.send_json(io, {"type": "bye"})
                logger.info("[service] shutdown requested")
                return False
            elif command == "cancel":
                continue
            else:
                MotionStreamProtocol.send_json(
                    io, {"type": "error", "message": f"unknown cmd {command!r}"}
                )

    def serve(
        self,
        host: str,
        port: int,
        idle_seconds: int = 600,
    ) -> None:
        server = socket.create_server((host, port))
        server.settimeout(idle_seconds)
        backbone = "no model" if self.pipeline is None else self.pipeline.backbone
        logger.info(f"[service] ready on {host}:{port} ({backbone}, idle-exit {idle_seconds}s)")

        while True:
            try:
                conn, addr = server.accept()
            except TimeoutError:
                logger.info("[service] no client for idle window -> exiting (frees the GPU)")
                return

            logger.info(f"[service] client connected {addr}")
            io = conn.makefile("rwb")
            try:
                if not self.serve_client(conn, io):
                    return
            except (ConnectionError, BrokenPipeError, json.JSONDecodeError):
                logger.info("[service] client disconnected")
            finally:
                io.close()
                conn.close()
