from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from text2motion.dataset.contracts import AMASS_LABEL_FPS
from text2motion.dataset.core.rate import cubic_resample
from text2motion.dataset.sources.amass.extractor import AmassPoseExtractor
from text2motion.dataset.sources.amass.names import _FOLDER_NAMES, DATASET_ALIASES
from text2motion.dataset.sources.amass.trim import head_trim_frames
from text2motion.motion.representation import JOINTS

EXCLUDED_AMASS_FOLDERS = frozenset({"MOYO"})


def source_parts(source_path: str) -> list[str]:
    return [part for part in Path(source_path).parts if part not in (".", "pose_data")]


def pose_folders(source_path: str, pose_root: Path) -> list[Path]:
    parts = source_parts(source_path)
    if not parts:
        return []
    datasets = list(dict.fromkeys([_FOLDER_NAMES.get(parts[0], parts[0]), parts[0]]))
    return list(
        dict.fromkeys(
            folder
            for dataset in datasets
            for folder in (
                pose_root / Path(dataset, *parts[1:-1]),
                pose_root / Path(dataset, dataset, *parts[1:-1]),
                pose_root / Path(dataset, parts[0], *parts[1:-1]),
            )
        )
    )


def pose_candidates(source_path: str, pose_root: Path) -> list[Path]:
    parts = source_parts(source_path)
    if not parts:
        return []
    stem = Path(parts[-1]).stem
    base = stem[:-6] if stem.endswith("_poses") else stem
    stems = list(dict.fromkeys([base, base.replace(" ", "_"), stem, stem.replace(" ", "_")]))
    names = list(
        dict.fromkeys(
            f"{candidate}{suffix}.npy"
            for candidate in stems
            for suffix in ("_stageii", "_poses", "")
        )
    )
    return [folder / name for folder in pose_folders(source_path, pose_root) for name in names]


def resolve_pose_path(source_path: str, pose_root: Path) -> Path | None:
    candidates = pose_candidates(source_path, pose_root)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def amass_capture_key(path: str) -> str:
    parts = [
        part
        for part in re.split(r"[\\/]", path.strip().lstrip("./"))
        if part and part != "pose_data"
    ]
    dataset = DATASET_ALIASES.get(parts[0].lower(), parts[0].lower())
    stem = re.sub(r"_(poses|stageii)$", "", Path(parts[-1]).stem.lower())
    subject = parts[-2].lower() if len(parts) > 2 else ""
    return f"{dataset}/{subject.replace(' ', '_')}/{stem.replace(' ', '_')}"


def hml3d_capture_keys(hml3d_dir: Path) -> frozenset[str]:
    with open(hml3d_dir / "index.csv", newline="", encoding="utf-8") as handle:
        return frozenset(
            amass_capture_key(row["source_path"])
            for row in csv.DictReader(handle)
            if "humanact12" not in row["source_path"].lower()
        )


def amass_files_outside_hml3d(raw_root: Path, hml3d_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in raw_root.rglob("*_stageii.npz")
        if not EXCLUDED_AMASS_FOLDERS.intersection(path.relative_to(raw_root).parts)
    )
    if not files:
        raise FileNotFoundError(f"no raw AMASS files under {raw_root}")
    used = hml3d_capture_keys(hml3d_dir)
    return [
        path
        for path in files
        if amass_capture_key(path.relative_to(raw_root).as_posix()) not in used
    ]


############################################################
#       AmassReader
############################################################
class AmassReader:
    def __init__(self, smplx_models: Path, target_fps: int) -> None:
        self._target_fps = target_fps
        self._at_target = AmassPoseExtractor(smplx_models, joint_count=JOINTS, fps=target_fps)
        self._at_label = AmassPoseExtractor(smplx_models, joint_count=JOINTS, fps=AMASS_LABEL_FPS)

    def raw_joints(self, path: Path, stored_fps: int = AMASS_LABEL_FPS) -> np.ndarray | None:
        if stored_fps == AMASS_LABEL_FPS:
            return self._at_target.extract(path)
        joints = self._at_label.extract(path)
        if joints is None:
            return None
        return cubic_resample(joints, stored_fps, self._target_fps)

    def clip_joints(self, path: Path, raw_root: Path) -> np.ndarray | None:
        joints = self.raw_joints(path)
        if joints is None:
            return None
        relative = path.relative_to(raw_root).as_posix()
        joints = joints[head_trim_frames(relative, self._target_fps) :].copy()
        joints[..., 0] *= -1
        return joints
