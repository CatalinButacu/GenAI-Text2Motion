import socket
import threading

import numpy as np

from text2motion.streaming.protocol import MotionInferenceClient, MotionStreamProtocol


def _stub_server(server: socket.socket, script: str, chunks: list[np.ndarray]) -> None:
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
            MotionStreamProtocol.send_array(io, "features", np.zeros((len(chunk), 263), dtype=np.float32))
        MotionStreamProtocol.send_json(io, {"type": "done", "frames": sum(len(c) for c in chunks)})
    elif script == "error":
        MotionStreamProtocol.send_json(io, {"type": "error", "message": "boom"})
    io.close()
    conn.close()
    server.close()


def _start(script: str, chunks: list[np.ndarray]) -> int:
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    thread = threading.Thread(target=_stub_server, args=(server, script, chunks), daemon=True)
    thread.start()
    return port


def test_client_streams_chunks_and_done():
    chunks = [np.random.rand(4, 22, 3).astype(np.float32) for _ in range(3)]
    port = _start("ok", chunks)
    client = MotionInferenceClient("127.0.0.1", port)
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
    port = _start("error", [])
    client = MotionInferenceClient("127.0.0.1", port)
    try:
        list(client.generate("walk", steps=3, temperature=1.0, top_p=0.9, cfg_scale=1.0))
        raise AssertionError("expected RuntimeError from service error message")
    except RuntimeError as exc:
        assert "boom" in str(exc)
    client.close()
