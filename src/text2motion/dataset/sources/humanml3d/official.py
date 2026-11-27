from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from text2motion.dataset.sources.base import HML3D_FPS
from text2motion.dataset.sources.humanml3d.source import check_hml3d_fps
from text2motion.motion.contracts import Split
from text2motion.motion.normalization import fit_train_stats
from text2motion.motion.representation import (
    DIM,
    JOINTS,
)
from text2motion.motion.storage import (
    FEATURES_DIR,
    atomic_numpy_save,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

OFFICIAL_SPLIT_FILES = ("train.txt", "val.txt", "test.txt")


def import_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "the official-release converter needs pyarrow: run uv sync --extra prep"
        ) from exc
    return pq


def checked_release_motion(name: str, motion) -> np.ndarray:
    feature = np.asarray(motion, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[0] == 0 or feature.shape[-1] != DIM:
        raise ValueError(f"{name}: expected [frames, {DIM}], got {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError(f"{name}: the motion holds NaN or Inf values")
    return feature


def index_frame_scale(fps: int) -> int:
    check_hml3d_fps(fps)
    return fps // HML3D_FPS


def parquet_files_of_split(src: Path, split: str) -> list[Path]:
    matches = sorted(src.glob(f"**/{split}-*.parquet"))
    if not matches:
        matches = sorted(p for p in src.glob("**/*.parquet") if split in p.name)
    return matches


def write_batch_motions(rows: dict, split: str, vec_dir: Path, seen: dict[str, str]) -> list[str]:
    names = []
    for motion, meta in zip(rows["motion"], rows["meta_data"], strict=True):
        name = meta["name"]
        if name in seen:
            raise ValueError(
                f"{name} appears in {seen[name]!r} and again in {split!r}; "
                "one feature file cannot belong to two splits"
            )
        seen[name] = split
        atomic_numpy_save(checked_release_motion(name, motion), vec_dir / f"{name}.npy")
        names.append(name)
    return names


def write_parquet_file(
    pq, parquet_file: Path, split: str, vec_dir: Path, seen: dict[str, str]
) -> list[str]:
    reader = pq.ParquetFile(parquet_file)
    desc = f"{split}:{parquet_file.name}"
    names = []
    for batch in tqdm(reader.iter_batches(batch_size=256), desc=desc):
        names.extend(write_batch_motions(batch.to_pydict(), split, vec_dir, seen))
    return names


def write_split_motions(
    pq, src: Path, split: str, vec_dir: Path, seen: dict[str, str]
) -> list[str]:
    files = parquet_files_of_split(src, split)
    if not files:
        raise FileNotFoundError(f"no parquet files for split {split!r} under {src}")
    names = []
    for parquet_file in files:
        names.extend(write_parquet_file(pq, parquet_file, split, vec_dir, seen))
    return names


def write_motions(src: Path, out_dir: Path) -> dict[str, list[str]]:
    pq = import_parquet()

    vec_dir = out_dir / FEATURES_DIR
    vec_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    split_names: dict[str, list[str]] = {
        split: write_split_motions(pq, src, split, vec_dir, seen) for split in Split
    }

    for split, names in split_names.items():
        (out_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    return split_names


def convert_official_release(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    split_names = write_motions(src, out)
    for split, names in split_names.items():
        logger.info(f"{split}: {len(names)} clips")
    fit_train_stats(out / FEATURES_DIR, split_names[Split.TRAIN], out, JOINTS)
