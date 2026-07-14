import socket
import threading

import numpy as np

from text2motion.stream.service import MotionServiceClient, read_msg, write_array, write_msg


def _stub_server(server: socket.socket, script: str, chunks: list[np.ndarray]) -> None:
    conn, _ = server.accept()
    io = conn.makefile("rwb")
    write_msg(io, {"type": "hello", "ckpt": "stub.pt", "backbone": "transformer", "downsample": 4})
    req = read_msg(io)
    assert req["cmd"] == "generate"
    if script == "ok":
        for chunk in chunks:
            write_array(io, "chunk", chunk)
        write_msg(io, {"type": "done", "frames": sum(len(c) for c in chunks)})
    elif script == "error":
        write_msg(io, {"type": "error", "message": "boom"})
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
    client = MotionServiceClient("127.0.0.1", port)
    assert client.hello["downsample"] == 4
    received = list(client.generate("walk", steps=3, temperature=1.0, top_p=0.9, cfg_scale=1.0))
    assert len(received) == 3
    for got, sent in zip(received, chunks):
        np.testing.assert_allclose(got, sent)
        assert got.dtype == np.float32
        assert got.shape == (4, 22, 3)
    client.close()


def test_client_raises_on_service_error():
    port = _start("error", [])
    client = MotionServiceClient("127.0.0.1", port)
    try:
        list(client.generate("walk", steps=3, temperature=1.0, top_p=0.9, cfg_scale=1.0))
        raise AssertionError("expected RuntimeError from service error message")
    except RuntimeError as exc:
        assert "boom" in str(exc)
    client.close()
