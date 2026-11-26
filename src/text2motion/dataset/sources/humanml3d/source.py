from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from text2motion.dataset.core.encoding import feature_joints
from text2motion.dataset.core.rate import cubic_resample
from text2motion.dataset.sources.base import (
    HML3D_FPS,
    INDEX_MANIFEST,
    MIRROR_PREFIXES,
    RecordingSource,
)
from text2motion.dataset.splits.plan import SplitPlan


class HumanML3DSource(RecordingSource):
    name = "humanml3d"

    def __init__(self, official_dir: Path) -> None:
        self.official_dir = Path(official_dir)
        self._plan = SplitPlan.from_official(self.official_dir)

    def list_recordings(self, raw_dir: Path) -> list[Path]:
        return []

    def split_of(self, clip: str) -> str | None:
        return self._plan.split_of.get(clip)

    def fill_missing(
        self, out_dir: Path, joints_dir: Path, fps: int, splits: tuple[str, ...] = ("train",)
    ) -> int:
        manifest = json.loads((Path(out_dir) / INDEX_MANIFEST).read_text(encoding="utf-8"))
        missing = [str(name).removesuffix(".npy") for name in manifest["missing_names"]]
        listed = set().union(*(self._plan.names(split) for split in splits))
        names = [name for name in missing if name in listed]
        vectors = self.official_dir / "new_joint_vecs"
        absent = [name for name in names if not (vectors / f"{name}.npy").is_file()]
        if absent:
            raise FileNotFoundError(f"{len(absent)} clips have no official features: {absent[:3]}")
        filled = [name for name in names if np.load(vectors / f"{name}.npy").shape[0] >= 2]
        for name in tqdm(filled, desc="fill from official"):
            for prefix in MIRROR_PREFIXES:
                features = np.load(vectors / f"{prefix}{name}.npy").astype(np.float32)
                clip = cubic_resample(feature_joints(features), HML3D_FPS, fps)
                np.save(Path(joints_dir) / f"{prefix}{name}.npy", clip)
        return len(filled)


def reference_joints_path(joints_dir: Path, clip_id: str) -> Path:
    preferred = joints_dir / f"{clip_id}.npy"
    if not preferred.is_file():
        raise FileNotFoundError(
            f"{preferred} missing; the skeleton reference is clip {clip_id}, "
            "and another clip changes every bone length"
        )
    return preferred


def check_hml3d_fps(fps: int) -> None:
    if fps <= 0 or fps % HML3D_FPS != 0:
        raise ValueError(
            f"the HumanML3D index frames are {HML3D_FPS} fps; need a multiple, got fps={fps}"
        )
