import socket
import threading
from typing import cast

import numpy as np

from text2motion.app.service.client import MotionInferenceClient
from text2motion.app.service.protocol import MotionStreamProtocol, ServiceAddress


def stub_server(server: socket.socket, script: str, chunks: list[np.ndarray]) -> None:
    conn, _ = server.accept()
    io = conn.makefile("rwb")
    MotionStreamProtocol.send_json(
        io,
        {
            "type": "hello",
            "ckpt": "stub.pt",
            "backbone": "transformer",
            "downsample": 4,
            "trained_steps": 49,
            "max_steps": 96,
            "device": "cpu",
        },
    )
    req = MotionStreamProtocol.recv_json(io)
    assert req["cmd"] == "generate"
    if script == "ok":
        for index, chunk in enumerate(chunks):
            MotionStreamProtocol.send_array(io, "chunk", chunk)
            MotionStreamProtocol.send_json(io, {"type": "placement", "yaw": 0.25 * index})
            MotionStreamProtocol.send_array(
                io, "features", np.zeros((len(chunk), 263), dtype=np.float32)
            )
        MotionStreamProtocol.send_json(io, {"type": "done", "frames": sum(len(c) for c in chunks)})
    elif script == "error":
        MotionStreamProtocol.send_json(io, {"type": "error", "message": "boom"})
    io.close()
    conn.close()
    server.close()


def start(script: str, chunks: list[np.ndarray]) -> int:
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    thread = threading.Thread(target=stub_server, args=(server, script, chunks), daemon=True)
    thread.start()
    return port


def test_client_streams_chunks_and_done():
    chunks = [np.random.rand(4, 22, 3).astype(np.float32) for _ in range(3)]
    port = start("ok", chunks)
    client = MotionInferenceClient(ServiceAddress("127.0.0.1", port))
    assert client.hello["downsample"] == 4
    assert client.hello["trained_steps"] == 49
    received = list(client.generate("walk", steps=3, temperature=1.0, top_p=0.9, cfg_scale=1.0))
    assert len(received) == 3
    for index, (got, sent) in enumerate(zip(received, chunks, strict=True)):
        np.testing.assert_allclose(got.joints, sent)
        assert got.joints.dtype == np.float32
        assert got.joints.shape == (4, 22, 3)
        assert got.features.shape == (4, 263)
        assert got.yaw == 0.25 * index
    client.close()


def test_client_raises_on_service_error():
    port = start("error", [])
    client = MotionInferenceClient(ServiceAddress("127.0.0.1", port))
    try:
        list(client.generate("walk", steps=3, temperature=1.0, top_p=0.9, cfg_scale=1.0))
        raise AssertionError("expected RuntimeError from service error message")
    except RuntimeError as exc:
        assert "boom" in str(exc)
    client.close()


class _HelloOnlyPipeline:
    def hello(self) -> dict:
        return {"type": "hello", "ckpt": "stub.pt", "backbone": "mamba"}


def serve_in_thread(pipeline) -> tuple[socket.socket, object, threading.Thread]:
    from text2motion.app.service.handlers import MotionInferenceServer

    server_side, client_side = socket.socketpair()
    server = MotionInferenceServer(pipeline, policy={"anchor_every_tokens": 0})
    server_io = server_side.makefile("rwb")
    thread = threading.Thread(
        target=server.serve_client, args=(server_side, server_io), daemon=True
    )
    thread.start()
    client_side.settimeout(5)
    return client_side, client_side.makefile("rwb"), thread


def test_a_cancel_that_arrives_after_the_stream_ended_gets_no_reply():
    client_side, io, thread = serve_in_thread(_HelloOnlyPipeline())
    assert MotionStreamProtocol.recv_json(io)["type"] == "hello"
    MotionStreamProtocol.send_json(io, {"cmd": "cancel"})
    MotionStreamProtocol.send_json(io, {"cmd": "shutdown"})
    assert MotionStreamProtocol.recv_json(io)["type"] == "bye"
    thread.join(timeout=5)
    client_side.close()


def test_a_client_connecting_after_a_failed_load_gets_an_empty_hello():
    client_side, io, thread = serve_in_thread(None)
    hello = MotionStreamProtocol.recv_json(io)
    assert hello["type"] == "hello"
    assert hello["ckpt"] == ""
    assert hello["policy"] == {"anchor_every_tokens": 0}
    MotionStreamProtocol.send_json(io, {"cmd": "shutdown"})
    assert MotionStreamProtocol.recv_json(io)["type"] == "bye"
    thread.join(timeout=5)
    client_side.close()


class _EndlessPipeline:
    device = "cpu"
    downsample = 4
    backbone = "mamba"
    checkpoint = "stub.pt"
    anchor_every = 0
    anchor_context = 16

    def __init__(self) -> None:
        from types import SimpleNamespace

        import torch

        def endless_tokens(*args, **kwargs):
            while True:
                yield torch.zeros(1, 2, dtype=torch.long)

        def chunks(token_iter):
            for _ in token_iter:
                yield torch.zeros(1, 16, 263)

        self.generator = SimpleNamespace(
            text_encoder=lambda prompts: torch.zeros(1, 512),
            token_generator=SimpleNamespace(stream_token_indices=endless_tokens),
        )
        self.decoder = SimpleNamespace(stream_tokens=chunks)

    def hello(self) -> dict:
        return {"type": "hello", "ckpt": self.checkpoint, "backbone": self.backbone}


def test_a_command_sent_mid_stream_cancels_the_stream_and_is_then_handled():
    client_side, io, thread = serve_in_thread(_EndlessPipeline())
    assert MotionStreamProtocol.recv_json(io)["type"] == "hello"
    MotionStreamProtocol.send_json(
        io,
        {
            "cmd": "generate",
            "prompt": "walk",
            "steps": 0,
            "temperature": 1.0,
            "top_p": 0.9,
            "cfg_scale": 1.0,
        },
    )
    header = MotionStreamProtocol.recv_json(io)
    assert header["type"] == "chunk"
    MotionStreamProtocol.recv_array(io, header)
    MotionStreamProtocol.send_json(io, {"cmd": "shutdown"})
    kinds = []
    while True:
        msg = MotionStreamProtocol.recv_json(io)
        kinds.append(msg["type"])
        if msg["type"] in ("chunk", "features"):
            MotionStreamProtocol.recv_array(io, msg)
        if msg["type"] in ("cancelled", "done", "bye", "error"):
            if msg["type"] != "bye":
                continue
            break
    assert "cancelled" in kinds
    assert kinds[-1] == "bye"
    thread.join(timeout=5)
    client_side.close()


def abandon_server(server: socket.socket) -> None:
    conn, _ = server.accept()
    io = conn.makefile("rwb")
    MotionStreamProtocol.send_json(io, {"type": "hello", "ckpt": "stub.pt", "backbone": "mamba"})
    for _ in range(2):
        request = MotionStreamProtocol.recv_json(io)
        assert request["cmd"] == "generate"
        cancelled = False
        for _ in range(50):
            MotionStreamProtocol.send_array(io, "chunk", np.zeros((4, 22, 3), np.float32))
            MotionStreamProtocol.send_json(io, {"type": "placement", "yaw": 0.0})
            MotionStreamProtocol.send_array(io, "features", np.zeros((4, 263), np.float32))
            conn.setblocking(False)
            try:
                peeked = conn.recv(1, socket.MSG_PEEK)
            except BlockingIOError:
                peeked = b""
            conn.setblocking(True)
            if peeked:
                assert MotionStreamProtocol.recv_json(io)["cmd"] == "cancel"
                cancelled = True
                break
        MotionStreamProtocol.send_json(
            io, {"type": "cancelled" if cancelled else "done", "frames": 0}
        )
    io.close()
    conn.close()
    server.close()


def test_a_client_that_stops_reading_early_leaves_the_connection_usable():
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    threading.Thread(target=abandon_server, args=(server,), daemon=True).start()
    client = MotionInferenceClient(ServiceAddress("127.0.0.1", port))
    client.sock.settimeout(5)
    stream = client.generate("walk", steps=0, temperature=1.0, top_p=0.9, cfg_scale=1.0)
    next(stream)
    stream.close()
    received = list(client.generate("walk", steps=0, temperature=1.0, top_p=0.9, cfg_scale=1.0))
    assert len(received) == 50
    client.close()


def test_inference_stats_report_the_time_to_the_first_chunk():
    from text2motion.app.service.handlers import MotionInferenceServer
    from text2motion.app.service.state import LoadedInferencePipeline

    pipeline = cast(LoadedInferencePipeline, _HelloOnlyPipeline())
    server = MotionInferenceServer(pipeline, policy={})
    timing = {"generate": 0.4, "send": 0.1, "tokens": 10, "first_chunk": 0.25}

    stats = server.inference_stats(1.0, 0.05, timing, 40, False)

    assert stats["first_chunk_ms"] == 250.0
    assert stats["ms_per_step"] == 40.0


class _SteerablePipeline(_EndlessPipeline):
    anchor_every = 4

    def __init__(self) -> None:
        import time
        from types import SimpleNamespace

        import torch

        def text_encoder(prompts):
            return torch.full((1, 512), 5.0 if prompts[0] == "run" else 0.0)

        def tokens(text, steps, sampling, anchors):
            index = 0
            while index < steps:
                if index and index % anchors.every == 0:
                    text = anchors.text_at(index)
                time.sleep(0.01)
                yield torch.full((1, 2), int(text[0, 0].item()), dtype=torch.long)
                index += 1

        def chunks(token_iter):
            for token in token_iter:
                yield torch.full((1, 4, 263), float(token[0, 0].item()))

        self.generator = SimpleNamespace(
            text_encoder=text_encoder, token_generator=SimpleNamespace(stream_token_indices=tokens)
        )
        self.decoder = SimpleNamespace(stream_tokens=chunks)


def test_steer_during_a_stream_keeps_the_stream_and_reports_the_joint_jump_in_cm():
    client_side, io, thread = serve_in_thread(_SteerablePipeline())
    assert MotionStreamProtocol.recv_json(io)["type"] == "hello"
    MotionStreamProtocol.send_json(
        io,
        {
            "cmd": "generate",
            "prompt": "walk",
            "steps": 60,
            "temperature": 1.0,
            "top_p": 0.9,
            "cfg_scale": 1.0,
        },
    )
    header = MotionStreamProtocol.recv_json(io)
    assert header["type"] == "chunk"
    MotionStreamProtocol.recv_array(io, header)
    MotionStreamProtocol.send_json(io, {"cmd": "steer", "prompt": "run"})
    kinds = []
    steered = None
    frames = 0
    while True:
        msg = MotionStreamProtocol.recv_json(io)
        kinds.append(msg["type"])
        if msg["type"] == "chunk":
            frames += len(MotionStreamProtocol.recv_array(io, msg))
        elif msg["type"] == "features":
            MotionStreamProtocol.recv_array(io, msg)
        elif msg["type"] == "steered":
            steered = msg
        elif msg["type"] in ("done", "cancelled", "error"):
            last = msg
            break
    assert last["type"] == "done"
    assert "cancelled" not in kinds
    assert kinds.count("steer_accepted") == 1
    assert steered is not None
    assert steered["prompt"] == "run"
    assert steered["switch_token"] % 4 == 0
    assert steered["frames_measured"] > 0
    assert steered["joint_jump_cm"] > 0
    assert last["frames"] == 60 * 4
    assert frames + 4 == 60 * 4
    MotionStreamProtocol.send_json(io, {"cmd": "shutdown"})
    assert MotionStreamProtocol.recv_json(io)["type"] == "bye"
    thread.join(timeout=5)
    client_side.close()


def test_steer_without_a_running_stream_is_rejected():
    client_side, io, thread = serve_in_thread(_HelloOnlyPipeline())
    assert MotionStreamProtocol.recv_json(io)["type"] == "hello"
    MotionStreamProtocol.send_json(io, {"cmd": "steer", "prompt": "run"})
    assert MotionStreamProtocol.recv_json(io)["type"] == "steer_rejected"
    MotionStreamProtocol.send_json(io, {"cmd": "shutdown"})
    assert MotionStreamProtocol.recv_json(io)["type"] == "bye"
    thread.join(timeout=5)
    client_side.close()
