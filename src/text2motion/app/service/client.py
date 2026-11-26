from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path

import numpy as np

from text2motion.app.service.protocol import MotionStreamProtocol, Release, ServiceAddress
from text2motion.app.service.spawn import (
    motion_server_command,
    motion_server_env,
    spawn_detached_process,
)
from text2motion.motion.contracts import StreamedChunk

logger = logging.getLogger(__name__)


############################################################
#       MotionInferenceClient
############################################################
class MotionInferenceClient:
    def __init__(self, address: ServiceAddress, timeout_seconds: float = 5.0):
        self.sock = socket.create_connection((address.host, address.port), timeout=timeout_seconds)
        self.sock.settimeout(None)
        self.io = self.sock.makefile("rwb")
        self.lock = threading.Lock()
        self.hello = MotionStreamProtocol.recv_json(self.io)
        self.last_stats: dict | None = None
        self.steer_events: list[dict] = []
        self.send_lock = threading.Lock()

    def steer(self, prompt: str) -> None:
        with self.send_lock:
            MotionStreamProtocol.send_json(self.io, {"cmd": "steer", "prompt": prompt})

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
            self.send_generate_request(prompt, steps, temperature, top_p, cfg_scale)
            yield from self.stream_generation_frames(should_cancel)

    def send_generate_request(
        self,
        prompt: str,
        steps: int,
        temperature: float,
        top_p: float,
        cfg_scale: float,
    ) -> None:
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

    def stream_generation_frames(self, should_cancel):
        cancel_sent = False
        finished = False
        try:
            while True:
                msg = MotionStreamProtocol.recv_json(self.io)
                kind = msg["type"]
                if kind == "chunk":
                    joints, placement, features = self.receive_chunk_frames(msg)
                    cancel_sent = self.send_cancel_once(should_cancel, cancel_sent)
                    yield StreamedChunk(
                        joints=joints, features=features, yaw=float(placement["yaw"])
                    )
                elif kind in ("steer_accepted", "steer_rejected", "steered"):
                    self.steer_events.append(msg)
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
                self.abandon_stream(cancel_sent)

    def receive_chunk_frames(self, msg: dict) -> tuple[np.ndarray, dict, np.ndarray]:
        joints = MotionStreamProtocol.recv_array(self.io, msg)
        placement = MotionStreamProtocol.recv_json(self.io)
        features = MotionStreamProtocol.recv_array(self.io, MotionStreamProtocol.recv_json(self.io))
        return joints, placement, features

    def send_cancel_once(self, should_cancel, cancel_sent: bool) -> bool:
        if should_cancel is None or not should_cancel() or cancel_sent:
            return cancel_sent
        with self.send_lock:
            MotionStreamProtocol.send_json(self.io, {"cmd": "cancel"})
        return True

    def abandon_stream(self, cancel_sent: bool) -> None:
        if not cancel_sent:
            MotionStreamProtocol.send_json(self.io, {"cmd": "cancel"})
        while True:
            msg = MotionStreamProtocol.recv_json(self.io)
            kind = msg["type"]
            if kind in ("chunk", "features"):
                MotionStreamProtocol.recv_array(self.io, msg)
            elif kind in ("steer_accepted", "steer_rejected", "steered"):
                continue
            elif kind in ("done", "cancelled", "error"):
                self.last_stats = msg.get("stats")
                return

    def close(self) -> None:
        try:
            self.io.close()
        finally:
            self.sock.close()


def start_motion_server(
    release: Release,
    address: ServiceAddress,
    log_path: str | Path,
    service_config: str | Path,
) -> None:
    cmd = motion_server_command(release, address, service_config)
    spawn_detached_process(cmd, motion_server_env(), log_path)
    logger.info(f"[service] spawned; loading the model (log: {log_path})...")


def await_motion_server(
    address: ServiceAddress, wait_seconds: int, log_path: str | Path
) -> MotionInferenceClient:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            return MotionInferenceClient(address)
        except OSError:
            time.sleep(2)
    raise RuntimeError(
        f"motion service not up on {address.host}:{address.port} after {wait_seconds}s; "
        f"see {log_path}"
    )


def changed_identity_fields(served: object, expected: dict[str, str]) -> list[str]:
    if not isinstance(served, dict):
        return sorted(expected)
    return sorted(key for key in expected if served.get(key) != expected[key])


def connect_or_spawn_motion_server(
    release: Release,
    address: ServiceAddress,
    log_path: str | Path,
    service_config: str | Path,
    spawn_wait_seconds: int,
) -> MotionInferenceClient:
    try:
        client = MotionInferenceClient(address)
        logger.info(f"[service] reusing warm service ({client.hello.get('ckpt')})")
    except OSError:
        start_motion_server(release, address, log_path, service_config)
        client = await_motion_server(address, spawn_wait_seconds, log_path)
    return client


def check_served_policy(
    client: MotionInferenceClient,
    address: ServiceAddress,
    service_config: str | Path,
    policy: dict[str, object],
) -> None:
    served_policy = client.hello.get("policy")
    if served_policy != policy:
        client.close()
        raise RuntimeError(
            f"the warm service on {address.host}:{address.port} runs the policy {served_policy}, "
            f"but {service_config} asks for {policy}; a load cannot change it. "
            "Stop the warm service, then start again."
        )


def switch_release_if_changed(
    client: MotionInferenceClient, release: Release
) -> MotionInferenceClient:
    expected = release.identity()
    changed = changed_identity_fields(client.hello.get("identity"), expected)
    if not changed:
        return client
    logger.info(f"[service] switching warm service to {release.ckpt} (different: {changed})")
    client.load(release.config, release.ckpt, release.tokenizer_ckpt, release.backbone)
    still_changed = changed_identity_fields(client.hello.get("identity"), expected)
    if still_changed:
        raise RuntimeError(f"the service loaded another release (different: {still_changed})")
    return client


def motion_service_client(
    release: Release,
    address: ServiceAddress,
    log_path: str | Path,
    *,
    service_config: str | Path,
    policy: dict[str, object],
    spawn_wait_seconds: int = 240,
) -> MotionInferenceClient:
    client = connect_or_spawn_motion_server(
        release, address, log_path, service_config, spawn_wait_seconds
    )
    check_served_policy(client, address, service_config, policy)
    return switch_release_if_changed(client, release)
