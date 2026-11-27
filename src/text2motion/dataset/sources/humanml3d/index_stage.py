from __future__ import annotations

import csv
import logging
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from text2motion.common.manifest import write_manifest
from text2motion.dataset.core.windows import CropError, cut_mirrored_window
from text2motion.dataset.sources.amass.reader import resolve_pose_path
from text2motion.dataset.sources.amass.trim import head_trim_frames
from text2motion.dataset.sources.base import INDEX_MANIFEST
from text2motion.dataset.sources.humanml3d.layout import Hml3dPreparationLayout
from text2motion.dataset.sources.humanml3d.official import index_frame_scale
from text2motion.motion.contracts import PreparationRequest
from text2motion.motion.representation import swap_left_right
from text2motion.motion.storage import atomic_numpy_save, sha256_file

logger = logging.getLogger(__name__)

DROPPED_INDEX_LOG = "dropped_index_rows.csv"


def cut_index_row(
    row: Mapping[Hashable, Any], pose_root: Path, fps: int, scale: int
) -> np.ndarray | None:
    source_path = row["source_path"]
    start_frame = int(row["start_frame"]) * scale
    end_frame = int(row["end_frame"]) * scale

    load_path = resolve_pose_path(source_path, pose_root)
    if load_path is None or ("humanact12" in source_path and scale != 1):
        return None
    data = np.load(load_path)

    source_is_prepared = "humanact12" in source_path
    trim = 0 if source_is_prepared else head_trim_frames(source_path, fps)
    return cut_mirrored_window(
        data, start_frame, end_frame, trim, source_is_prepared=source_is_prepared
    )


def write_index_clips(
    rows: Sequence[Mapping[Hashable, Any]], joints_dir: Path, pose_root: Path, fps: int, scale: int
) -> tuple[int, list[str], list[tuple[str, str, str]]]:
    written = 0
    missing: list[str] = []
    invalid: list[tuple[str, str, str]] = []
    for row in tqdm(rows, desc="index->joints"):
        new_name = row["new_name"]
        try:
            data = cut_index_row(row, pose_root, fps, scale)
        except CropError as error:
            invalid.append((str(new_name), row["source_path"], str(error)))
            continue
        if data is None:
            missing.append(str(new_name))
            continue

        data_m = swap_left_right(data)
        atomic_numpy_save(data, joints_dir / f"{new_name}")
        atomic_numpy_save(data_m, joints_dir / f"M{new_name}")
        written += 1
    return written, missing, invalid


def write_index_manifest(
    out_dir: Path,
    recipe: dict[str, object],
    written: int,
    missing: list[str],
    invalid: list[tuple[str, str, str]],
) -> None:
    status = {
        "written": written,
        "missing": len(missing),
        "missing_names": missing,
        "invalid": len(invalid),
        "invalid_names": [name for name, _, _ in invalid],
        "complete": not missing and not invalid,
    }
    write_manifest(out_dir / INDEX_MANIFEST, recipe, status)


def write_dropped_rows(out_dir: Path, invalid: list[tuple[str, str, str]]) -> None:
    with open(out_dir / DROPPED_INDEX_LOG, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["new_name", "source_path", "reason"])
        writer.writerows(invalid)


def log_index_outcome(
    written: int, missing: list[str], invalid: list[tuple[str, str, str]], total: int
) -> None:
    if invalid:
        logger.warning(
            f"index->joints: dropped {len(invalid)} rows with invalid crop bounds or too few "
            f"frames (log: {DROPPED_INDEX_LOG}); first: {[name for name, _, _ in invalid[:3]]}"
        )
    if missing:
        logger.warning(
            f"index->joints: {len(missing)} of {total} clips have no pose file "
            f"(names in {INDEX_MANIFEST}); first: {missing[:3]}"
        )
    logger.info(f"index->joints: wrote {written} clips")


def build_indexed_joint_clips(
    request: PreparationRequest, layout: Hml3dPreparationLayout, pose_root: Path | None = None
) -> None:
    if request.index_csv is None:
        raise ValueError("index_csv must be set for stage 'index'")
    fps = request.fps
    scale = index_frame_scale(fps)
    import pandas as pd

    index_file = pd.read_csv(request.index_csv)
    pose_root = pose_root if pose_root is not None else layout.pose_data

    written, missing, invalid = write_index_clips(
        index_file.to_dict("records"), layout.joints, pose_root, fps, scale
    )

    recipe = {"index_sha256": sha256_file(Path(request.index_csv)), "fps": fps}
    write_index_manifest(layout.out_dir, recipe, written, missing, invalid)
    write_dropped_rows(layout.out_dir, invalid)
    log_index_outcome(written, missing, invalid, len(index_file))
