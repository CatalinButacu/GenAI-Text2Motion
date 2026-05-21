"""Single-entry seed helper for reproducible training and inference.

Use ``seed_all(42)`` at the top of any script that touches randomness.
Covers: Python `random`, NumPy, PyTorch (CPU + CUDA), cuDNN, and the
``PYTHONHASHSEED`` env var (informational — must be set before interp start).

Two modes:

  seed_all(42)                    # fast: seeds RNGs only.
  seed_all(42, deterministic=True) # strict: also enables deterministic CUDA
                                  # ops + sets CUBLAS_WORKSPACE_CONFIG.
                                  # ~5-20% slower; use for benchmarks and
                                  # for the runs that back published numbers.
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_all(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that the training/inference pipeline can touch.

    Args:
        seed: integer seed used for every RNG. Use a different seed per run
            for ablations; reuse the same seed when comparing two
            implementations of the same model.
        deterministic: when True, also enables ``torch.use_deterministic_algorithms``
            and sets ``CUBLAS_WORKSPACE_CONFIG``. Trades ~5-20 % throughput
            for byte-identical reproducibility across runs on the same
            hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if deterministic:
        # PYTHONHASHSEED is consumed by the Python interpreter at startup;
        # setting it here only affects subprocesses we spawn.
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only: don't crash if a layer doesn't have a deterministic impl;
        # just emit a warning so the user can decide.
        torch.use_deterministic_algorithms(True, warn_only=True)
    expected = str(seed)
    actual = os.environ.get("PYTHONHASHSEED")

    if deterministic and actual is not None and actual != expected:
        warnings.warn(
            f"PYTHONHASHSEED was already set to {actual!r} before seed_all() ran; "
            f"requested {expected!r}. Hash-dependent dict iteration order will "
            f"not match other runs unless you export PYTHONHASHSEED before "
            f"starting Python.",
            stacklevel=2,
        )
    log.info(
        "[seed] seed_all(seed=%d, deterministic=%s) applied to python+numpy+torch%s",
        seed,
        deterministic,
        " (+cudnn deterministic, CUBLAS_WORKSPACE_CONFIG)" if deterministic else "",
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding hook. Pass as ``worker_init_fn=seed_worker``.

    PyTorch sets ``torch.initial_seed()`` per worker; we mirror that to
    NumPy + ``random`` so augmentations in worker subprocesses are
    deterministic too (still distinct across workers).
    """
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def seed_dict(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Return a dict describing the seed config — log to wandb or save to JSON."""
    return {
        "seed": seed,
        "deterministic": deterministic,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
