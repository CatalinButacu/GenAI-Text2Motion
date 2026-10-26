from __future__ import annotations

import gc
import json
import logging
import select
import socket
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.representation import FPS, StreamingSkeletonRecovery
from text2motion.streaming.decoder import StreamingMotionDecoder
from text2motion.streaming.protocol import MotionStreamProtocol

logger = logging.getLogger(__name__)


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


_InferencePipelineLoader = Callable[[dict], LoadedInferencePipeline]


def pending_request(sock: socket.socket, io) -> dict | None:
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return None
    return MotionStreamProtocol.recv_json(io)


class MotionInferenceServer:
    def __init__(
        self,
        pipeline: LoadedInferencePipeline,
        pipeline_loader: _InferencePipelineLoader | None = None,
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
