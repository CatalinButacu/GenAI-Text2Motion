# .\.venv\Scripts\python.exe scripts/extra_motion/quality_forest.py
import csv
import pathlib

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from text2motion.motion.representation import recover_from_ric
from text2motion.motion.storage import base_clip_ids

REPO = pathlib.Path(__file__).resolve().parents[2]
HML3D = REPO / "data" / "HumanML3D_official"
FEATURES = REPO / "data" / "extra_motion" / "features20"
CLIP_CSV = REPO / "data" / "extra_motion" / "quality_clips.csv"
SUMMARY_CSV = REPO / "doc" / "notes" / "quality_forest.csv"
FPS = 20
JOINT_COUNT = 22
FOOT_JOINTS = (10, 11)
CONTACT_HEIGHT_M = 0.05
SKID_M_PER_FRAME = 0.025
FLOAT_TOLERANCE_M = 0.005
METRES_TO_MM = 1000.0
YAW_PERCENTILE = 99
MIN_FRAMES = 40
SEED = 2026
TREES = 100
METRICS = ("skating_ratio", "float_mm", "jerk_m_per_s3", "yaw_p99_rad")
EXTRA_SOURCES = ("amass_extra", "motionx_lp6", "motionx_official_lp6")
CLIP_COLUMNS = ("source", "subset", "name", "seconds", *METRICS, "inlier")
SUMMARY_COLUMNS = ("source", "subset", "clips", "hours", "inlier_share", "hours_kept")


def __clip_metrics(features: np.ndarray) -> tuple[float, ...]:
    joints = recover_from_ric(torch.from_numpy(features).float(), JOINT_COUNT).numpy()
    joints = joints.astype(np.float64)
    feet = joints[:, FOOT_JOINTS]
    contact = feet[:-1, :, 1] < CONTACT_HEIGHT_M
    skid = np.linalg.norm(np.diff(feet[..., [0, 2]], axis=0), axis=-1) > SKID_M_PER_FRAME
    skating = float((contact & skid).any(axis=-1).mean())
    lowest = feet[..., 1].min(axis=-1)
    floating = float(np.maximum(0.0, lowest - FLOAT_TOLERANCE_M).mean() * METRES_TO_MM)
    jerk = np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1) * FPS**3
    yaw = float(np.percentile(np.abs(features[:, 0]), YAW_PERCENTILE))
    return skating, floating, float(jerk.mean()), yaw


def __hml3d_paths(split: str) -> list[pathlib.Path]:
    paths = [HML3D / "new_joint_vecs" / f"{name}.npy" for name in base_clip_ids(HML3D, split)]
    return [path for path in paths if path.is_file()]


def __measure(source: str, paths: list[pathlib.Path]) -> list[dict[str, float | str | bool]]:
    rows = []
    short = 0
    for path in paths:
        features = np.load(path).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"{path}: non-finite features")
        if features.shape[0] < MIN_FRAMES:
            short += 1
            continue
        values = __clip_metrics(features)
        row = {
            "source": source,
            "subset": path.stem.split("__")[0] if "__" in path.stem else source,
            "name": path.stem,
            "seconds": features.shape[0] / FPS,
        }
        row.update(zip(METRICS, values, strict=True))
        rows.append(row)
    print(source, "clips", len(rows), "shorter than", MIN_FRAMES, "frames", short, flush=True)
    return rows


def __matrix(rows: list[dict[str, float | str | bool]]) -> np.ndarray:
    return np.array([[float(row[key]) for key in METRICS] for row in rows])


def __summary(rows: list[dict[str, float | str | bool]]) -> list[tuple[object, ...]]:
    groups: dict[tuple[str, str], list[dict[str, float | str | bool]]] = {}
    for row in rows:
        groups.setdefault((str(row["source"]), "all"), []).append(row)
        if row["subset"] != row["source"]:
            groups.setdefault((str(row["source"]), str(row["subset"])), []).append(row)
    table = []
    for (source, subset), members in sorted(groups.items()):
        seconds = np.array([float(row["seconds"]) for row in members])
        kept = np.array([bool(row["inlier"]) for row in members])
        table.append(
            (
                source,
                subset,
                len(members),
                round(seconds.sum() / 3600, 2),
                round(float(kept.mean()), 3),
                round(seconds[kept].sum() / 3600, 2),
            )
        )
    return table


def main() -> None:
    train = __measure("hml3d_train", __hml3d_paths("train"))
    rows = train + __measure("hml3d_val", __hml3d_paths("val"))
    for source in EXTRA_SOURCES:
        rows += __measure(source, sorted((FEATURES / source).glob("*.npy")))
    forest = IsolationForest(n_estimators=TREES, contamination="auto", random_state=SEED)
    forest.fit(__matrix(train))
    inlier = forest.predict(__matrix(rows)) == 1
    for row, keep in zip(rows, inlier, strict=True):
        row["inlier"] = bool(keep)
    with open(CLIP_CSV, "w", newline="", encoding="utf-8") as handle:
        clip_writer = csv.writer(handle, lineterminator="\n")
        clip_writer.writerow(CLIP_COLUMNS)
        clip_writer.writerows([row[column] for column in CLIP_COLUMNS] for row in rows)
    table = __summary(rows)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(table)
    for line in table:
        print(dict(zip(SUMMARY_COLUMNS, line, strict=True)))


if __name__ == "__main__":
    main()
