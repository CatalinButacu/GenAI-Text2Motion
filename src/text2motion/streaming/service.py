from __future__ import annotations

import gc
import json
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


def _process_memory_gb() -> tuple[float, float]:
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
        f"detokenize {stats['detokenize_ms']:.0f}ms | "
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

    def hello(self) -> dict:
        return {
            "type": "hello",
            "ckpt": self.checkpoint,
            "backbone": self.backbone,
            "downsample": self.downsample,
            "trained_steps": self.trained_steps,
            "max_steps": self.max_steps,
            "device": self.device,
        }


_InferencePipelineLoader = Callable[[dict], LoadedInferencePipeline]


def cancel_requested(sock: socket.socket, io) -> bool:
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return False
    return MotionStreamProtocol.recv_json(io).get("cmd") == "cancel"


class MotionInferenceServer:
    def __init__(
        self,
        pipeline: LoadedInferencePipeline,
        pipeline_loader: _InferencePipelineLoader | None = None,
        inference_log: str | Path | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.pipeline_loader = pipeline_loader
        self.inference_log = Path(inference_log) if inference_log else None

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
        decode_seconds = 0.0
        encode_started = time.perf_counter()
        with torch.no_grad():
            text_emb = self.pipeline.generator.text_encoder([request["prompt"]]).to(self.pipeline.device)
            encode_seconds = time.perf_counter() - encode_started
            token_iter = self.pipeline.generator.token_generator.stream_token_indices(
                text_emb,
                int(request["steps"]),
                temperature=float(request["temperature"]),
                top_p=float(request["top_p"]),
                cfg_scale=float(request["cfg_scale"]),
                stop_at_end=False,
            )
            for chunk in self.pipeline.decoder.stream_tokens(token_iter):
                chunk_started = time.perf_counter()
                features = chunk.squeeze(0).cpu().numpy()
                carried_yaw = recovery.yaw
                joints = recovery(features)
                decode_seconds += time.perf_counter() - chunk_started
                frames += len(joints)
                MotionStreamProtocol.send_array(io, "chunk", joints)
                MotionStreamProtocol.send_json(io, {"type": "placement", "yaw": carried_yaw})
                MotionStreamProtocol.send_array(io, "features", features)
                if cancel_requested(sock, io):
                    cancelled = True
                    break

        elapsed = time.perf_counter() - started
        print(
            f"[service] {'cancelled' if cancelled else 'done'} {frames} frames in {elapsed:.2f}s "
            f"({request['prompt']!r})",
            flush=True,
        )
        stats = self.inference_stats(elapsed, encode_seconds, decode_seconds, frames, on_cuda)
        print(f"[perf] {format_stats(stats)}", flush=True)
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
        self, elapsed: float, encode: float, decode: float, frames: int, on_cuda: bool
    ) -> dict:
        steps = max(1, frames // max(1, self.pipeline.downsample))
        host_rss, host_peak = _process_memory_gb()
        generate = max(0.0, elapsed - encode - decode)
        return {
            "frames": frames,
            "steps": steps,
            "total_ms": elapsed * 1e3,
            "text_encode_ms": encode * 1e3,
            "generate_ms": generate * 1e3,
            "detokenize_ms": decode * 1e3,
            "ms_per_step": generate / steps * 1e3,
            "realtime_factor": (frames / FPS) / elapsed if elapsed > 0 else 0.0,
            "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if on_cuda else 0.0,
            "host_rss_gb": host_rss,
            "host_peak_gb": host_peak,
        }

    def handle_load(self, io, request: dict) -> None:
        if self.pipeline_loader is None:
            MotionStreamProtocol.send_json(io, {"type": "error", "message": "this service cannot reload models"})
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
        print(f"[service] loaded {self.pipeline.checkpoint} ({self.pipeline.backbone})", flush=True)
        MotionStreamProtocol.send_json(io, {**self.pipeline.hello(), "type": "loaded"})

    def serve_client(self, conn: socket.socket, io) -> bool:
        MotionStreamProtocol.send_json(io, self.pipeline.hello())
        while True:
            request = MotionStreamProtocol.recv_json(io)
            command = request.get("cmd")

            if command == "generate":
                try:
                    self.handle_generate(conn, io, request)
                except (ConnectionError, BrokenPipeError):
                    raise
                except Exception as exc:
                    traceback.print_exc()
                    MotionStreamProtocol.send_json(io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            elif command == "load":
                try:
                    self.handle_load(io, request)
                except Exception as exc:
                    traceback.print_exc()
                    MotionStreamProtocol.send_json(io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            elif command == "shutdown":
                MotionStreamProtocol.send_json(io, {"type": "bye"})
                print("[service] shutdown requested", flush=True)
                return False
            else:
                MotionStreamProtocol.send_json(io, {"type": "error", "message": f"unknown cmd {command!r}"})

    def serve(
        self,
        host: str,
        port: int,
        idle_seconds: int = 600,
    ) -> None:
        server = socket.create_server((host, port))
        server.settimeout(idle_seconds)
        print(
            f"[service] ready on {host}:{port} ({self.pipeline.backbone}, idle-exit {idle_seconds}s)",
            flush=True,
        )

        while True:
            try:
                conn, addr = server.accept()
            except TimeoutError:
                print("[service] no client for idle window -> exiting (frees the GPU)", flush=True)
                return

            print(f"[service] client connected {addr}", flush=True)
            io = conn.makefile("rwb")
            try:
                if not self.serve_client(conn, io):
                    return
            except (ConnectionError, BrokenPipeError, json.JSONDecodeError):
                print("[service] client disconnected", flush=True)
            finally:
                io.close()
                conn.close()
