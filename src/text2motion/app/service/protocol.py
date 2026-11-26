from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text2motion.motion.storage import sha256_file

logger = logging.getLogger(__name__)


############################################################
#       MotionStreamProtocol
############################################################
class MotionStreamProtocol:
    @staticmethod
    def send_json(io, obj: dict) -> None:
        io.write((json.dumps(obj) + "\n").encode("utf-8"))
        io.flush()

    @staticmethod
    def recv_json(io) -> dict:
        line = io.readline()
        if not line:
            raise ConnectionError("peer closed")
        return json.loads(line.decode("utf-8"))

    @classmethod
    def send_array(cls, io, kind: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        cls.send_json(io, {"type": kind, "shape": list(arr.shape), "dtype": "float32"})
        io.write(arr.tobytes())
        io.flush()

    @staticmethod
    def recv_array(io, msg: dict) -> np.ndarray:
        count = int(np.prod(msg["shape"]))
        data = io.read(count * 4)
        if not data or len(data) != count * 4:
            raise ConnectionError("short read")
        return np.frombuffer(data, dtype=np.float32).reshape(msg["shape"]).copy()


############################################################
#       Release
############################################################
@dataclass(frozen=True)
class Release:
    config: str
    ckpt: str
    tokenizer_ckpt: str
    backbone: str

    def identity(self) -> dict[str, str]:
        return {
            "backbone": str(self.backbone),
            "config": str(Path(self.config).resolve()),
            "config_sha256": sha256_file(self.config),
            "ckpt": str(Path(self.ckpt).resolve()),
            "ckpt_sha256": sha256_file(self.ckpt),
            "tokenizer": str(Path(self.tokenizer_ckpt).resolve()),
            "tokenizer_sha256": sha256_file(self.tokenizer_ckpt),
        }


############################################################
#       ServiceAddress
############################################################
@dataclass(frozen=True)
class ServiceAddress:
    host: str
    port: int
