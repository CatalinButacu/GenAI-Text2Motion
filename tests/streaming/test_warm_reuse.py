import socket
import threading
from types import MappingProxyType

import pytest

from text2motion.streaming.protocol import (
    MotionStreamProtocol,
    motion_service_client,
    release_identity,
)
from text2motion.studio.scene import StudioSession

POLICY = MappingProxyType(
    {"anchor_every_tokens": 0, "anchor_context_tokens": 16, "cuda_graph_decode": False}
)


def __release(tmp_path) -> tuple[str, str, str, str]:
    paths = {name: tmp_path / f"{name}.bin" for name in ("config", "ckpt", "tokenizer")}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    return str(paths["config"]), str(paths["ckpt"]), str(paths["tokenizer"]), "mamba"


def __warm_service(hello: dict, loaded_identity: dict, commands: list[str]) -> int:
    server = socket.create_server(("127.0.0.1", 0))

    def serve() -> None:
        conn, _ = server.accept()
        io = conn.makefile("rwb")
        MotionStreamProtocol.send_json(io, hello)
        try:
            request = MotionStreamProtocol.recv_json(io)
        except (ConnectionError, ValueError):
            request = {}
        commands.append(str(request.get("cmd")))
        if request.get("cmd") == "load":
            reply = {**hello, "type": "loaded", "identity": loaded_identity}
            MotionStreamProtocol.send_json(io, reply)
        io.close()
        conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1]


def __connect(release: tuple[str, str, str, str], port: int, tmp_path):
    return motion_service_client(
        *release,
        "127.0.0.1",
        port,
        tmp_path / "service.log",
        service_config="service.yml",
        policy=dict(POLICY),
    )


def test_a_warm_service_with_the_same_release_is_reused(tmp_path):
    release = __release(tmp_path)
    identity = release_identity(*release)
    commands: list[str] = []
    port = __warm_service(
        {"type": "hello", "identity": identity, "policy": dict(POLICY)}, {}, commands
    )

    client = __connect(release, port, tmp_path)

    assert client.hello["identity"] == identity
    client.close()


def test_a_retrained_checkpoint_at_the_same_path_is_reloaded(tmp_path):
    release = __release(tmp_path)
    old_identity = release_identity(*release)
    (tmp_path / "ckpt.bin").write_bytes(b"retrained")
    new_identity = release_identity(*release)
    commands: list[str] = []
    hello = {"type": "hello", "identity": old_identity, "policy": dict(POLICY)}
    port = __warm_service(hello, new_identity, commands)

    client = __connect(release, port, tmp_path)

    assert commands == ["load"]
    assert client.hello["identity"]["ckpt_sha256"] == new_identity["ckpt_sha256"]
    client.close()


def test_a_service_that_loads_another_release_is_refused(tmp_path):
    release = __release(tmp_path)
    identity = release_identity(*release)
    commands: list[str] = []
    hello = {"type": "hello", "identity": {}, "policy": dict(POLICY)}
    port = __warm_service(hello, {**identity, "tokenizer_sha256": "other"}, commands)

    with pytest.raises(RuntimeError, match="tokenizer_sha256"):
        __connect(release, port, tmp_path)


def test_a_warm_service_with_another_policy_is_refused(tmp_path):
    release = __release(tmp_path)
    identity = release_identity(*release)
    commands: list[str] = []
    other_policy = {**POLICY, "anchor_every_tokens": 48}
    port = __warm_service(
        {"type": "hello", "identity": identity, "policy": other_policy}, {}, commands
    )

    with pytest.raises(RuntimeError, match="Stop the warm service"):
        __connect(release, port, tmp_path)


def test_a_model_change_refreshes_every_derived_limit():
    session = StudioSession.__new__(StudioSession)
    session.apply_model_limits({"downsample": 4, "trained_steps": 49, "max_steps": 96})

    session.apply_model_limits({"downsample": 4, "trained_steps": 30, "max_steps": 0})

    assert session.trained_steps == 30
    assert session.max_steps == 0
    assert session.default_steps == 30
