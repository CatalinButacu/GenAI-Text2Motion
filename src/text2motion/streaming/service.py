from __future__ import annotations

import json
import select
import socket
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import torch

from text2motion.generation.pipeline import MotionGenerator
from text2motion.motion.representation import recover_skeleton
from text2motion.streaming.decoder import StreamingMotionDecoder
from text2motion.streaming.protocol import Wire


@dataclass(frozen=True)
class ServiceModel:
    generator: MotionGenerator
    decoder: StreamingMotionDecoder
    backbone: str
    checkpoint: str
    downsample: int
    max_steps: int
    device: str

    def hello(self) -> dict:
        return {
            "type": "hello",
            "ckpt": self.checkpoint,
            "backbone": self.backbone,
            "downsample": self.downsample,
            "max_steps": self.max_steps,
            "device": self.device,
        }


ModelLoader = Callable[[dict], ServiceModel]


def cancel_requested(sock: socket.socket, io) -> bool:
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return False
    return Wire.recv_json(io).get("cmd") == "cancel"


class StreamingService:
    def __init__(self, model: ServiceModel, loader: ModelLoader | None = None) -> None:
        self.model = model
        self.loader = loader

    def handle_generate(self, sock: socket.socket, io, request: dict) -> None:
        frames = 0
        cancelled = False
        started = time.perf_counter()

        with torch.no_grad():
            token_iter = self.model.generator.module.stream(
                self.model.generator.text_encoder([request["prompt"]]).to(self.model.device),
                int(request["steps"]),
                temperature=float(request["temperature"]),
                top_p=float(request["top_p"]),
                cfg_scale=float(request["cfg_scale"]),
                stop_at_end=False,
            )
            for chunk in self.model.decoder.stream_tokens(token_iter):
                joints = recover_skeleton(chunk.squeeze(0).cpu().numpy())
                frames += len(joints)
                Wire.send_array(io, "chunk", joints)
                if cancel_requested(sock, io):
                    cancelled = True
                    break

        elapsed = time.perf_counter() - started
        print(
            f"[service] {'cancelled' if cancelled else 'done'} {frames} frames in {elapsed:.2f}s "
            f"({request['prompt']!r})",
            flush=True,
        )
        Wire.send_json(io, {"type": "cancelled" if cancelled else "done", "frames": frames})

    def handle_load(self, io, request: dict) -> None:
        if self.loader is None:
            Wire.send_json(io, {"type": "error", "message": "this service cannot reload models"})
            return
        self.model = self.loader(request)
        if self.model.device == "cuda":
            torch.cuda.empty_cache()
        print(f"[service] loaded {self.model.checkpoint} ({self.model.backbone})", flush=True)
        Wire.send_json(io, {**self.model.hello(), "type": "loaded"})

    def serve_client(self, conn: socket.socket, io) -> bool:
        Wire.send_json(io, self.model.hello())
        while True:
            request = Wire.recv_json(io)
            command = request.get("cmd")

            if command == "generate":
                try:
                    self.handle_generate(conn, io, request)
                except (ConnectionError, BrokenPipeError):
                    raise
                except Exception as exc:
                    traceback.print_exc()
                    Wire.send_json(io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            elif command == "load":
                try:
                    self.handle_load(io, request)
                except Exception as exc:
                    traceback.print_exc()
                    Wire.send_json(io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            elif command == "shutdown":
                Wire.send_json(io, {"type": "bye"})
                print("[service] shutdown requested", flush=True)
                return False
            else:
                Wire.send_json(io, {"type": "error", "message": f"unknown cmd {command!r}"})

    def serve(
        self,
        host: str = Wire.HOST,
        port: int = Wire.PORT,
        idle_seconds: int = 600,
    ) -> None:
        server = socket.create_server((host, port))
        server.settimeout(idle_seconds)
        print(
            f"[service] ready on {host}:{port} ({self.model.backbone}, idle-exit {idle_seconds}s)",
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
