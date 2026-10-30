from __future__ import annotations

import hashlib
import json
import logging
import os
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from text2motion.motion.contracts import Split

FEATURES_DIR = "new_joint_vecs"
TEXTS_DIR = "texts"
MEAN_FILE = "Mean.npy"
STD_FILE = "Std.npy"
STATS_PROVENANCE_FILE = "stats_provenance.json"
CLIP_LENGTHS_FILE = "clip_lengths.json"
RESUME_VERSION = 2
FAILURE_DUMP_FILE = "failure_dump.pt"

logger = logging.getLogger(__name__)


SPLIT_FILES = MappingProxyType({split: f"{split.value}.txt" for split in Split})


def split_clip_ids(root: Path, split: Split | str) -> list[str]:
    split_path = Path(root) / SPLIT_FILES[Split(split)]
    if not split_path.is_file():
        raise FileNotFoundError(
            f"split file not found: {split_path} (copy the official {split} list there)"
        )
    listed = [name.strip() for name in split_path.read_text(encoding="utf-8").splitlines()]
    return list(dict.fromkeys(name.removeprefix("M") for name in listed if name))


def atomic_torch_save(state: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_numpy_save(array: np.ndarray, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {"torch": torch.get_rng_state(), "cuda": cuda_states}


def non_finite_names(module: torch.nn.Module) -> list[str]:
    names = []
    for name, parameter in module.named_parameters():
        if not torch.isfinite(parameter).all():
            names.append(name)
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            names.append(f"{name}.grad")
    return names


def write_failure_dump(
    path: str | Path | None,
    state: dict[str, Any],
    inputs: dict[str, Any],
    epoch_step: int,
    module: torch.nn.Module,
    error: BaseException,
) -> None:
    if path is None:
        return
    dump = state | {
        "inputs": inputs,
        "epoch_step": epoch_step,
        "non_finite": non_finite_names(module),
        "error": "".join(traceback.format_exception(error)),
    }
    atomic_torch_save(dump, path)
    logger.error(f"wrote a failure dump to {path}; non-finite tensors: {dump['non_finite']}")


@contextmanager
def dump_on_failure(
    path: str | Path | None,
    capture: Callable[[], tuple[dict[str, Any], dict[str, Any], int, torch.nn.Module]],
) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        if path is not None:
            write_failure_dump(path, *capture(), error)
        raise


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if not state["cuda"]:
        return
    if not torch.cuda.is_available():
        raise ValueError("the resume file has a CUDA RNG state, but CUDA is not available")
    torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


def is_legacy_resume(state: dict[str, Any], path: Path, kind: str, accept_legacy: bool) -> bool:
    version = state.get("version", 1)
    if version == RESUME_VERSION and state.get("format") == kind:
        return False
    if version == RESUME_VERSION:
        raise ValueError(f"{path} is a {state.get('format')!r} resume file, not {kind!r}")
    if version != 1:
        raise ValueError(f"{path}: resume version {version} is not supported")
    if not accept_legacy:
        raise ValueError(
            f"{path} is an old resume file (version 1) without the RNG and schedule state; "
            "pass --accept_legacy_resume to load it anyway"
        )
    return True


def item_rng(seed: int, epoch: int, index: int) -> np.random.RandomState:
    item_seed = np.random.SeedSequence((seed, epoch, index)).generate_state(1)[0]
    return np.random.RandomState(item_seed)


def __read_frame_count_cache(cache_path: Path) -> dict[str, list[int]]:
    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {key: list(value) for key, value in raw.items()}


def clip_frame_counts(vec_dir: Path, clip_ids: list[str]) -> dict[str, int]:
    cache_path = vec_dir.parent / CLIP_LENGTHS_FILE
    cache = __read_frame_count_cache(cache_path)

    lengths: dict[str, int] = {}
    dirty = False
    for clip_id in clip_ids:
        path = vec_dir / f"{clip_id}.npy"
        if not path.is_file():
            continue
        stat = path.stat()
        identity = [stat.st_size, stat.st_mtime_ns]
        entry = cache.get(clip_id)
        if entry is not None and entry[1:] == identity:
            lengths[clip_id] = entry[0]
            continue
        length = int(np.load(path, mmap_mode="r").shape[0])
        lengths[clip_id] = length
        cache[clip_id] = [length, *identity]
        dirty = True

    if dirty:
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(cache), encoding="utf-8")
        temporary.replace(cache_path)
    return lengths
