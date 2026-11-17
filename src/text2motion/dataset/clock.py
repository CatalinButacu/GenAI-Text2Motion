from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from text2motion.dataset.amass import dataset_name, head_trim_frames, resolve_pose_path
from text2motion.dataset.contracts import AMASS_LABEL_FPS
from text2motion.dataset.features import feature_joints
from text2motion.motion.representation import FPS

STORED_FPS_HYPOTHESES = (60, 100, 120)
PROBE_JOINTS = (0, 7, 8, 15, 20, 21)
PAIRS = tuple(itertools.combinations(PROBE_JOINTS, 2))
MIN_CORRELATION = 0.9
MIN_SCORED_FRAMES = 2
NO_SCORE = -1.0
USABLE_STATUS = frozenset({"ok", "folder_rate"})


############################################################
#       ClockRow
############################################################
@dataclass(frozen=True)
class ClockRow:
    source_path: str
    dataset: str
    clips: int
    correlations: tuple[float, ...]
    stored_fps: int
    margin: float
    status: str
    longest_official_frames: int

    def as_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "source_path": self.source_path,
            "dataset": self.dataset,
            "label_fps": AMASS_LABEL_FPS,
            "clips": self.clips,
        }
        for fps, value in zip(STORED_FPS_HYPOTHESES, self.correlations, strict=True):
            record[f"corr_{fps}"] = round(value, 4)
        record.update(stored_fps=self.stored_fps, margin=round(self.margin, 4), status=self.status)
        return record


def __distances(joints: np.ndarray) -> np.ndarray:
    return np.stack([np.linalg.norm(joints[:, a] - joints[:, b], axis=-1) for a, b in PAIRS], 1)


def __correlation(ours: np.ndarray, official: np.ndarray) -> float:
    a = ours - ours.mean(axis=0)
    b = official - official.mean(axis=0)
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def __clip_score(
    pose: np.ndarray, official: np.ndarray, first_frame: int, stored_fps: int, pose_stride: int
) -> float:
    seconds = (first_frame + np.arange(official.shape[0])) / FPS
    positions = seconds * stored_fps / pose_stride
    if positions[-1] > pose.shape[0] - 1:
        return float("nan")
    frames = np.arange(pose.shape[0])
    flat = pose.reshape(pose.shape[0], -1)
    columns = [np.interp(positions, frames, flat[:, c]) for c in range(flat.shape[1])]
    resampled = np.stack(columns, 1).reshape(len(positions), -1, 3)
    return __correlation(__distances(resampled), official)


def __file_row(source_path: str, scores: list[tuple[float, ...]], longest: int) -> ClockRow:
    table = np.array(scores)
    means = np.nanmean(np.where(np.isnan(table), NO_SCORE, table), axis=0)
    order = np.argsort(means)[::-1]
    best, second = int(order[0]), int(order[1])
    return ClockRow(
        source_path=source_path,
        dataset=dataset_name(source_path),
        clips=len(scores),
        correlations=tuple(float(value) for value in means),
        stored_fps=STORED_FPS_HYPOTHESES[best],
        margin=float(means[best] - means[second]),
        status="ok" if means[best] >= MIN_CORRELATION else "no_match",
        longest_official_frames=longest,
    )


def __folder(source_path: str) -> str:
    return source_path.rsplit("/", 1)[0]


def with_folder_rates(rows: list[ClockRow]) -> list[ClockRow]:
    rates: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row.status == "ok":
            rates[__folder(row.source_path)].add(row.stored_fps)
    updated = []
    for row in rows:
        folder_rates = rates[__folder(row.source_path)]
        unscored = row.longest_official_frames < MIN_SCORED_FRAMES
        if unscored and len(folder_rates) == 1:
            row = replace(row, stored_fps=next(iter(folder_rates)), status="folder_rate")
        updated.append(row)
    return updated


def stored_fps_table(official_dir: Path, pose_root: Path, pose_stride: int) -> list[ClockRow]:
    index = pd.read_csv(official_dir / "index.csv")
    scores: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    longest: dict[str, int] = defaultdict(int)
    for row in index.to_dict("records"):
        source_path = str(row["source_path"])
        name = str(row["new_name"]).removesuffix(".npy")
        features_path = official_dir / "new_joint_vecs" / f"{name}.npy"
        pose_path = resolve_pose_path(source_path, pose_root)
        if "humanact12" in source_path or pose_path is None or not features_path.is_file():
            continue
        official = __distances(feature_joints(np.load(features_path).astype(np.float32)))
        pose = np.load(pose_path)
        first = head_trim_frames(source_path, FPS) + int(row["start_frame"])
        scores[source_path].append(
            tuple(
                __clip_score(pose, official, first, fps, pose_stride)
                for fps in STORED_FPS_HYPOTHESES
            )
        )
        longest[source_path] = max(longest[source_path], official.shape[0])
    rows = [__file_row(path, scores[path], longest[path]) for path in sorted(scores)]
    return with_folder_rates(rows)
