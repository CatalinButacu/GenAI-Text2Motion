from __future__ import annotations

import gc
import json
import logging
import socket
import time
import traceback
from pathlib import Path

import torch

from text2motion.app.service.pipeline import InferencePipelineLoader, LoadedInferencePipeline
from text2motion.app.service.protocol import MotionStreamProtocol
from text2motion.app.service.stats import format_stats, process_memory_gb, record_inference
from text2motion.app.service.steering import StreamSteering
from text2motion.app.service.stream import (
    StreamRun,
    inference_record,
    pending_request,
    send_failure,
    send_no_model_error,
    start_stream,
    timed_tokens,
)
from text2motion.motion.representation import FPS

logger = logging.getLogger(__name__)


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

    def apply_steer(
        self, io, request: dict | None, steering: StreamSteering, token: int
    ) -> dict | None:
        if request is None or request.get("cmd") != "steer":
            return request
        if self.pipeline is None:
            raise RuntimeError("steer needs a loaded model")
        new_text = self.pipeline.generator.text_encoder([request["prompt"]]).to(
            self.pipeline.device
        )
        MotionStreamProtocol.send_json(io, steering.request(request["prompt"], new_text, token))
        return None

    def handle_generate(self, sock: socket.socket, io, request: dict) -> None:
        if self.pipeline is None:
            send_no_model_error(io)
            return
        pipeline = self.pipeline
        run = StreamRun(pipeline, pipeline.device == "cuda", time.perf_counter())
        if run.on_cuda:
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            steering, token_source = start_stream(pipeline, run, request)
            self.stream_chunks(sock, io, run, steering, token_source)
        for report in steering.flush():
            MotionStreamProtocol.send_json(io, report)
        self.finish_stream(io, run, request)

    def stream_chunks(
        self, sock: socket.socket, io, run: StreamRun, steering, token_source
    ) -> None:
        tokens = timed_tokens(token_source, run.timing, run.on_cuda)
        for chunk in run.pipeline.decoder.stream_tokens(tokens):
            features = chunk.squeeze(0).cpu().numpy()
            carried_yaw = run.recovery.yaw
            joints = run.recovery(features)
            run.frames += len(joints)
            send_started = time.perf_counter()
            MotionStreamProtocol.send_array(io, "chunk", joints)
            MotionStreamProtocol.send_json(io, {"type": "placement", "yaw": carried_yaw})
            MotionStreamProtocol.send_array(io, "features", features)
            run.timing.setdefault("first_chunk", time.perf_counter() - run.started)
            for report in steering.observe(joints):
                MotionStreamProtocol.send_json(io, report)
            interrupting = self.apply_steer(
                io, pending_request(sock, io), steering, run.timing["tokens"]
            )
            run.timing["send"] += time.perf_counter() - send_started
            run.cancelled = interrupting is not None
            if interrupting is not None and interrupting.get("cmd") != "cancel":
                logger.info(
                    f"[service] {interrupting.get('cmd')!r} arrived mid-stream -> stream cancelled, request deferred"
                )
                self.deferred_request = interrupting
            if run.cancelled:
                break

    def finish_stream(self, io, run: StreamRun, request: dict) -> None:
        elapsed = time.perf_counter() - run.started
        logger.info(
            f"[service] {'cancelled' if run.cancelled else 'done'} {run.frames} frames in {elapsed:.2f}s "
            f"({request['prompt']!r})"
        )
        stats = self.inference_stats(
            elapsed, run.encode_seconds, run.timing, run.frames, run.on_cuda
        )
        logger.info(f"[perf] {format_stats(stats)}")
        if self.inference_log is not None:
            record_inference(
                inference_record(run.pipeline, self.policy, request, run, stats),
                self.inference_log,
            )
        if run.on_cuda:
            torch.cuda.empty_cache()
        MotionStreamProtocol.send_json(
            io,
            {
                "type": "cancelled" if run.cancelled else "done",
                "frames": run.frames,
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
            if not self.dispatch_request(conn, io, request):
                return False

    def dispatch_request(self, conn: socket.socket, io, request: dict) -> bool:
        command = request.get("cmd")
        if command == "generate":
            try:
                self.handle_generate(conn, io, request)
            except (ConnectionError, BrokenPipeError):
                raise
            except Exception as exc:
                traceback.print_exc()
                send_failure(io, exc)
        elif command == "load":
            try:
                self.handle_load(io, request)
            except Exception as exc:
                traceback.print_exc()
                send_failure(io, exc)
        elif command == "shutdown":
            MotionStreamProtocol.send_json(io, {"type": "bye"})
            logger.info("[service] shutdown requested")
            return False
        elif command == "steer":
            MotionStreamProtocol.send_json(
                io, {"type": "steer_rejected", "reason": "no stream is running"}
            )
        elif command != "cancel":
            MotionStreamProtocol.send_json(
                io, {"type": "error", "message": f"unknown cmd {command!r}"}
            )
        return True

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
