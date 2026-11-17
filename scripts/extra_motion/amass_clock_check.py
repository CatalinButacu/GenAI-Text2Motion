# .\.venv\Scripts\python.exe scripts/extra_motion/amass_clock_check.py
import csv
import itertools
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from text2motion.motion.amass import dataset_name, head_trim_frames, resolve_pose_path
from text2motion.motion.representation import FPS, JOINTS, recover_from_ric

REPO = pathlib.Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "data" / "HumanML3D_official"
POSE_ROOT = REPO / "data" / "HumanML3D_60fps" / "pose_data"
OUT_CSV = REPO / "doc" / "notes" / "amass_clock_check.csv"
LABEL_FPS = 120
POSE_STRIDE = 2
STORED_FPS_HYPOTHESES = (60, 100, 120)
PROBE_JOINTS = (0, 7, 8, 15, 20, 21)
PAIRS = tuple(itertools.combinations(PROBE_JOINTS, 2))
MIN_CORRELATION = 0.9
MIN_SCORED_FRAMES = 2


def __distances(joints: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.linalg.norm(joints[:, a] - joints[:, b], axis=-1) for a, b in PAIRS], axis=1
    )


def __official_distances(name: str) -> np.ndarray:
    features = np.load(OFFICIAL / "new_joint_vecs" / f"{name}.npy").astype(np.float32)
    joints = recover_from_ric(torch.from_numpy(features), JOINTS).numpy()
    return __distances(joints)


def __correlation(ours: np.ndarray, official: np.ndarray) -> float:
    a = ours - ours.mean(axis=0)
    b = official - official.mean(axis=0)
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def __score(pose: np.ndarray, official: np.ndarray, first_frame: int, stored_fps: int) -> float:
    pose_fps = stored_fps / POSE_STRIDE
    seconds = (first_frame + np.arange(official.shape[0])) / FPS
    positions = seconds * pose_fps
    if positions[-1] > pose.shape[0] - 1:
        return float("nan")
    frames = np.arange(pose.shape[0])
    flat = pose.reshape(pose.shape[0], -1)
    resampled = np.stack(
        [np.interp(positions, frames, flat[:, c]) for c in range(flat.shape[1])], 1
    )
    return __correlation(__distances(resampled.reshape(-1, JOINTS, 3)), official)


def __folder(source_path: str) -> str:
    return source_path.rsplit("/", 1)[0]


def __apply_folder_rates(rows: list[dict], longest: dict[str, int]) -> None:
    rates: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row["status"] == "ok":
            rates[__folder(row["source_path"])].add(row["stored_fps"])
    for row in rows:
        folder_rates = rates[__folder(row["source_path"])]
        if longest[row["source_path"]] >= MIN_SCORED_FRAMES or len(folder_rates) != 1:
            continue
        row["stored_fps"] = next(iter(folder_rates))
        row["status"] = "folder_rate"


def main() -> None:
    index = pd.read_csv(OFFICIAL / "index.csv")
    per_file: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    longest: dict[str, int] = defaultdict(int)
    for row in index.to_dict("records"):
        source_path = str(row["source_path"])
        if "humanact12" in source_path:
            continue
        name = str(row["new_name"]).removesuffix(".npy")
        if not (OFFICIAL / "new_joint_vecs" / f"{name}.npy").is_file():
            continue
        pose_path = resolve_pose_path(source_path, POSE_ROOT)
        if pose_path is None:
            continue
        pose = np.load(pose_path)
        official = __official_distances(name)
        first_frame = head_trim_frames(source_path, FPS) + int(row["start_frame"])
        scores = tuple(
            __score(pose, official, first_frame, stored) for stored in STORED_FPS_HYPOTHESES
        )
        per_file[source_path].append(scores)
        longest[source_path] = max(longest[source_path], official.shape[0])

    rows = []
    for source_path, clip_scores in sorted(per_file.items()):
        table = np.array(clip_scores)
        means = np.nanmean(np.where(np.isnan(table), -1.0, table), axis=0)
        order = np.argsort(means)[::-1]
        best, second = int(order[0]), int(order[1])
        rows.append(
            {
                "source_path": source_path,
                "dataset": dataset_name(source_path),
                "label_fps": LABEL_FPS,
                "clips": len(clip_scores),
                **{
                    f"corr_{fps}": round(float(means[i]), 4)
                    for i, fps in enumerate(STORED_FPS_HYPOTHESES)
                },
                "stored_fps": STORED_FPS_HYPOTHESES[best],
                "margin": round(float(means[best] - means[second]), 4),
                "status": "ok" if means[best] >= MIN_CORRELATION else "no_match",
            }
        )
    __apply_folder_rates(rows, longest)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = pd.DataFrame(rows).groupby(["dataset", "stored_fps", "status"]).size()
    print(summary.to_string())
    print("files", len(rows), "->", OUT_CSV)


if __name__ == "__main__":
    main()
