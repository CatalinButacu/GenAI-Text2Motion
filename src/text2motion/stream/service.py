from __future__ import annotations

import argparse
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.joints import recover_skeleton
from text2motion.data.hml3d.stats import MotionScaler
from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import load_config
from text2motion.shared.seed import seed_everything
from text2motion.stream.decode import StreamingMotionDecoder

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def write_msg(io, obj: dict) -> None:
    io.write((json.dumps(obj) + "\n").encode("utf-8"))
    io.flush()


def read_msg(io) -> dict:
    line = io.readline()
    if not line:
        raise ConnectionError("peer closed")
    return json.loads(line.decode("utf-8"))


def write_array(io, kind: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    write_msg(io, {"type": kind, "shape": list(arr.shape), "dtype": "float32"})
    io.write(arr.tobytes())
    io.flush()


def read_exact(io, n_bytes: int) -> bytes:
    data = io.read(n_bytes)
    if data is None or len(data) != n_bytes:
        raise ConnectionError("short read")
    return data


def read_array(io, msg: dict) -> np.ndarray:
    count = int(np.prod(msg["shape"]))
    raw = read_exact(io, count * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(msg["shape"]).copy()


class Pipeline:
    def __init__(self, config: str, ckpt: str, tokenizer_ckpt: str, backbone: str, device: str):
        cfg = load_config(config)
        tok = ResidualFsqTokenizer(cfg.tokenizer)
        tok.load_state_dict(torch.load(tokenizer_ckpt, map_location="cpu"))
        self.tok = tok.to(device).eval()
        n_layers = cfg.generator.mamba_n_layers if backbone == "mamba" else cfg.generator.n_layers
        gen_cfg = replace(
            cfg.generator,
            backbone=backbone,
            n_layers=n_layers,
            num_codebooks=cfg.tokenizer.num_quantizers,
            codebook_size=self.tok.codebook_size,
            use_kernel=False,
        )
        gen = MotionGenerator(gen_cfg).to(device).eval()
        state = torch.load(ckpt, map_location="cpu")
        gen.load_state_dict(
            state["generator"] if isinstance(state, dict) and "generator" in state else state
        )
        self.gen = gen
        self.te = CLIPTextEncoder(cfg.text_encoder).to(device).eval()
        scaler = MotionScaler.load(Path(cfg.paths.hml3d_out_dir), dim=cfg.hml3d.dim)
        self.mean = torch.from_numpy(scaler.mean).to(device)
        self.std = torch.from_numpy(scaler.std).to(device)
        self.device = device
        self.downsample = int(cfg.tokenizer.downsample)
        self.max_steps = int(cfg.generator.max_seq_len + 1 - cfg.generator.text_prefix_len)
        self.decoder = StreamingMotionDecoder(self.tok, self.mean, self.std)  # probe once, not/req

        self.config = str(config)
        self.ckpt = str(ckpt)
        self.tokenizer_ckpt = str(tokenizer_ckpt)
        self.backbone = backbone

    def hello(self) -> dict:
        return {
            "type": "hello",
            "ckpt": self.ckpt,
            "backbone": self.backbone,
            "downsample": self.downsample,
            "max_steps": self.max_steps,
            "device": self.device,
        }


def cancel_requested(sock: socket.socket, io) -> bool:
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return False
    return read_msg(io).get("cmd") == "cancel"


def handle_generate(pipe: Pipeline, sock: socket.socket, io, req: dict) -> None:
    n_frames = 0
    cancelled = False
    started = time.perf_counter()
    with torch.no_grad():
        emb = pipe.te([req["prompt"]]).to(pipe.device)
        decoder = pipe.decoder
        token_iter = pipe.gen.stream(
            emb,
            int(req["steps"]),
            temperature=float(req["temperature"]),
            top_p=float(req["top_p"]),
            cfg_scale=float(req["cfg_scale"]),
            stop_at_end=False,
        )
        for chunk in decoder.stream_tokens(token_iter):
            joints = recover_skeleton(chunk.squeeze(0).cpu().numpy())
            n_frames += len(joints)
            write_array(io, "chunk", joints)
            if cancel_requested(sock, io):
                cancelled = True
                break
    elapsed = time.perf_counter() - started
    print(
        f"[service] {'cancelled' if cancelled else 'done'} {n_frames} frames in {elapsed:.2f}s "
        f"({req['prompt']!r})",
        flush=True,
    )
    write_msg(io, {"type": "cancelled" if cancelled else "done", "frames": n_frames})


def serve(pipeline_args: dict, host: str, port: int, idle_seconds: int) -> None:
    pipe = Pipeline(**pipeline_args)
    server = socket.create_server((host, port))
    server.settimeout(idle_seconds)
    print(
        f"[service] ready on {host}:{port} ({pipe.backbone}, idle-exit {idle_seconds}s)", flush=True
    )
    while True:
        try:
            conn, addr = server.accept()
        except TimeoutError:
            print("[service] no client for idle window -> exiting (frees the 4GB GPU)", flush=True)
            return
        print(f"[service] client connected {addr}", flush=True)
        io = conn.makefile("rwb")
        try:
            write_msg(io, pipe.hello())
            while True:
                req = read_msg(io)
                cmd = req.get("cmd")
                if cmd == "generate":
                    try:
                        handle_generate(pipe, conn, io, req)
                    except (ConnectionError, BrokenPipeError):
                        raise
                    except Exception as exc:
                        traceback.print_exc()
                        write_msg(io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                elif cmd == "load":
                    try:
                        pipe = Pipeline(
                            req["config"],
                            req["ckpt"],
                            req["tokenizer_ckpt"],
                            req["backbone"],
                            pipe.device,
                        )
                        if pipe.device == "cuda":
                            torch.cuda.empty_cache()
                        print(f"[service] loaded {pipe.ckpt} ({pipe.backbone})", flush=True)
                        write_msg(io, {**pipe.hello(), "type": "loaded"})
                    except Exception as exc:
                        traceback.print_exc()
                        write_msg(io, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                elif cmd == "shutdown":
                    write_msg(io, {"type": "bye"})
                    print("[service] shutdown requested", flush=True)
                    return
                else:
                    write_msg(io, {"type": "error", "message": f"unknown cmd {cmd!r}"})
        except (ConnectionError, BrokenPipeError, json.JSONDecodeError):
            print("[service] client disconnected", flush=True)
        finally:
            io.close()
            conn.close()


class MotionServiceClient:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(None)
        self.io = self.sock.makefile("rwb")
        self.lock = threading.Lock()
        self.hello = read_msg(self.io)

    def load(self, config: str, ckpt: str, tokenizer_ckpt: str, backbone: str) -> dict:
        with self.lock:
            write_msg(
                self.io,
                {
                    "cmd": "load",
                    "config": config,
                    "ckpt": ckpt,
                    "tokenizer_ckpt": tokenizer_ckpt,
                    "backbone": backbone,
                },
            )
            msg = read_msg(self.io)
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
            write_msg(
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
            cancel_sent = False
            while True:
                msg = read_msg(self.io)
                kind = msg["type"]
                if kind == "chunk":
                    joints = read_array(self.io, msg)
                    if should_cancel is not None and should_cancel() and not cancel_sent:
                        write_msg(self.io, {"cmd": "cancel"})
                        cancel_sent = True
                    yield joints
                elif kind in ("done", "cancelled"):
                    return
                elif kind == "error":
                    raise RuntimeError(msg["message"])
                else:
                    raise RuntimeError(f"unexpected message {kind!r}")

    def close(self) -> None:
        try:
            self.io.close()
        finally:
            self.sock.close()


def connect_or_spawn(
    config: str,
    ckpt: str,
    tokenizer_ckpt: str,
    backbone: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_path: str = "outputs/motion_service.log",
    spawn_wait_seconds: int = 240,
) -> MotionServiceClient:
    client = None
    try:
        client = MotionServiceClient(host, port)
        print(f"[service] reusing warm service ({client.hello.get('ckpt')})", flush=True)
    except OSError:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "text2motion.stream.service",
            "--config",
            config,
            "--ckpt",
            ckpt,
            "--tokenizer_ckpt",
            tokenizer_ckpt,
            "--backbone",
            backbone,
            "--host",
            host,
            "--port",
            str(port),
        ]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab")
        flags = getattr(subprocess, "DETACHED_PROCESS", 0)
        env = dict(os.environ)
        env["PYTHONPATH"] = env.get("PYTHONPATH") or "src"
        subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=flags, env=env)
        print(f"[service] spawned; loading the model (log: {log_path})...", flush=True)
        deadline = time.time() + spawn_wait_seconds
        while time.time() < deadline:
            try:
                client = MotionServiceClient(host, port)
                break
            except OSError:
                time.sleep(2)
        if client is None:
            raise RuntimeError(
                f"motion service not up on {host}:{port} after {spawn_wait_seconds}s; see {log_path}"
            )
    if Path(str(client.hello.get("ckpt", ""))).resolve() != Path(ckpt).resolve():
        print(f"[service] switching warm service to {ckpt}", flush=True)
        client.load(config, ckpt, tokenizer_ckpt, backbone)
    return client


def main() -> None:
    p = argparse.ArgumentParser(
        description="Local motion-generation service: loads the model once, streams joint chunks over TCP."
    )
    p.add_argument("--config", default="configs/generator/final100m_fsq8x1024.yaml")
    p.add_argument("--ckpt", default="checkpoints/generator/generator_transformer_100m_val163.pt")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--idle_minutes", type=int, default=10)
    a = p.parse_args()
    seed_everything(2026, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_args = {
        "config": a.config,
        "ckpt": a.ckpt,
        "tokenizer_ckpt": a.tokenizer_ckpt,
        "backbone": a.backbone,
        "device": device,
    }
    serve(pipeline_args, a.host, a.port, a.idle_minutes * 60)


if __name__ == "__main__":
    main()
