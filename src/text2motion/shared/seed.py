"""Reproducible seeding. Log the seed on every run -- the controlled twin (ADR 0001) must be
comparable across architectures, which requires fixed, recorded seeds. ``deterministic=True`` adds
strict bit-exact determinism (cuBLAS workspace + cuDNN deterministic + use_deterministic_algorithms)."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Must be set before the first cuBLAS call (we seed at the top of run(), before model build).
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            ":4096:8"  # deterministic cuBLAS matmul (CUDA >=10.2)
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # benchmark picks fastest (nondeterministic) kernels
        torch.use_deterministic_algorithms(True)  # raises if an op has no deterministic impl

    return seed
