from __future__ import annotations

import random

import numpy as np
import torch

from text2motion.app.config import ComputeDevice, Float32Precision


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def apply_precision(precision: Float32Precision | str) -> None:
    precision = Float32Precision(precision)
    allow_tf32 = precision is Float32Precision.TF32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def resolve_device(requested: ComputeDevice | str, override: str | None = None) -> str:
    if override:
        return override
    requested = ComputeDevice(requested)
    if requested is ComputeDevice.CUDA and not torch.cuda.is_available():
        return ComputeDevice.CPU.value
    return requested.value
