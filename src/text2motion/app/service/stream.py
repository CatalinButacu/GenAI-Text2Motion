from __future__ import annotations

import select
import socket
import time
from dataclasses import dataclass, field

import torch

from text2motion.app.service.protocol import MotionStreamProtocol
from text2motion.app.service.state import LoadedInferencePipeline, StreamSteering
from text2motion.generation.contracts import SamplingConfig, StreamAnchors
from text2motion.motion.representation import StreamingSkeletonRecovery


############################################################
#       StreamRun
############################################################
@dataclass
class StreamRun:
    pipeline: LoadedInferencePipeline
    on_cuda: bool
    started: float
    recovery: StreamingSkeletonRecovery = field(default_factory=StreamingSkeletonRecovery)
    timing: dict = field(default_factory=lambda: {"generate": 0.0, "send": 0.0, "tokens": 0})
    frames: int = 0
    cancelled: bool = False
    encode_seconds: float = 0.0


def pending_request(sock: socket.socket, io) -> dict | None:
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return None
    return MotionStreamProtocol.recv_json(io)


def synchronize_cuda(on_cuda: bool) -> None:
    if on_cuda:
        torch.cuda.synchronize()


def timed_tokens(token_source, timing: dict, on_cuda: bool):
    while True:
        token_started = time.perf_counter()
        tokens = next(token_source, None)
        synchronize_cuda(on_cuda)
        timing["generate"] += time.perf_counter() - token_started
        if tokens is None:
            return
        timing["tokens"] += 1
        yield tokens


def start_stream(pipeline: LoadedInferencePipeline, run: StreamRun, request: dict):
    encode_started = time.perf_counter()
    text_emb = pipeline.generator.text_encoder([request["prompt"]]).to(pipeline.device)
    synchronize_cuda(run.on_cuda)
    run.encode_seconds = time.perf_counter() - encode_started
    steering = StreamSteering(pipeline.anchor_every, pipeline.downsample, text_emb)
    token_source = pipeline.generator.token_generator.stream_token_indices(
        text_emb,
        int(request["steps"]),
        sampling=SamplingConfig(
            temperature=float(request["temperature"]),
            top_p=float(request["top_p"]),
            cfg_scale=float(request["cfg_scale"]),
            stop_at_end=False,
        ),
        anchors=StreamAnchors(
            every=pipeline.anchor_every,
            context=pipeline.anchor_context,
            text_at=steering.text_at if pipeline.anchor_every else None,
        ),
    )
    return steering, token_source


def inference_record(
    pipeline: LoadedInferencePipeline,
    policy: dict[str, object],
    request: dict,
    run: StreamRun,
    stats: dict,
) -> dict:
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": request["prompt"],
        "backbone": pipeline.backbone,
        "checkpoint": pipeline.checkpoint,
        "identity": pipeline.identity,
        "policy": policy,
        "device": pipeline.device,
        "cancelled": run.cancelled,
        "requested_steps": int(request["steps"]),
        "cfg_scale": float(request["cfg_scale"]),
        **stats,
    }


def send_no_model_error(io) -> None:
    MotionStreamProtocol.send_json(
        io,
        {
            "type": "error",
            "message": "no model loaded -- the last load failed and freed the previous "
            "one; pick a model and press Load model again",
        },
    )


def send_failure(io, exc: Exception) -> None:
    MotionStreamProtocol.send_json(io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
