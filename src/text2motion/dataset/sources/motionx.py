from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import numpy as np

from text2motion.dataset.contracts import MOTIONX_POINTS
from text2motion.motion.representation import JOINTS


def motionx_captions(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as archive:
        return {
            PurePosixPath(member).stem: archive.read(member).decode("utf-8").strip()
            for member in archive.namelist()
            if member.endswith(".txt")
        }


def body_joints(points: np.ndarray, member: str) -> np.ndarray:
    if points.ndim != 3 or points.shape[1:] != (MOTIONX_POINTS, 3):
        expected = f"(frames, {MOTIONX_POINTS}, 3)"
        raise ValueError(f"{member}: expected {expected}, got {points.shape}")
    joints = points[:, :JOINTS].astype(np.float32)
    if not np.isfinite(joints).all():
        raise ValueError(f"{member}: non-finite joints")
    return joints


def motionx_body_clips(zip_path: Path) -> Iterator[tuple[str, np.ndarray]]:
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(member for member in archive.namelist() if member.endswith(".npy"))
        for member in members:
            points = np.load(io.BytesIO(archive.read(member)))
            yield PurePosixPath(member).stem, body_joints(points, member)
