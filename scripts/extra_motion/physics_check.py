# .\.venv\Scripts\python.exe scripts/extra_motion/physics_check.py motionx
import csv
import pathlib
import sys

import numpy as np
import torch

from text2motion.evaluation.metrics import motion_statistics
from text2motion.motion.representation import FPS, recover_from_ric
from text2motion.motion.storage import base_clip_ids

REPO = pathlib.Path(__file__).resolve().parents[2]
EXTRA = REPO / "data" / "extra_motion"
HML3D = REPO / "data" / "HumanML3D_official"
JOINT_COUNT = 22
SKATE = "foot_skate_m_per_s"
JERK = "mean_jerk_cm_per_frame3"
ROOT_SPEED = "root_speed_p99_m_per_s"
ROOT_MEDIAN = "root_speed_median_m_per_s"
LOCOMOTION_SPEED = 0.5
MIN_GROUP_CLIPS = 30
GROUPS = ("locomotion", "other")
TREADMILL = "treadmill"
PLANTED_FOOT = "planted_foot_speed_median_m_per_s"
BELT_ROOT_SPEED = 0.3
BELT_FOOT_SPEED = 0.5
PLANTED_HEIGHT = 0.05
LEFT_FOOT = 10
FOOT_JOINTS = (10, 11)
SKATE_RATIO_LIMIT = 1.5
JERK_RATIO_LIMIT = 2.0
ROOT_SPEED_RATIO_LIMIT = 1.5
OUTLIER_PERCENTILE = 99
CLIP_COLUMNS = (
    "name",
    SKATE,
    JERK,
    ROOT_SPEED,
    "outlier",
    "group",
    "matched_outlier",
    "belt_walk",
)
SUMMARY_COLUMNS = (
    "source",
    "clips",
    "median_skate",
    "hml3d_median_skate",
    "skate_ratio",
    "median_jerk",
    "hml3d_median_jerk",
    "jerk_ratio",
    "root_speed_p99",
    "hml3d_root_speed_p99",
    "root_speed_ratio",
    "outlier_share",
    "pass_skate",
    "pass_jerk",
    "pass_root_speed",
)


def __clip_statistics(features: np.ndarray) -> dict[str, float]:
    joints = recover_from_ric(torch.from_numpy(features).float(), JOINT_COUNT).numpy()
    statistics = motion_statistics(joints)
    root_speed = np.linalg.norm(np.diff(joints[:, 0, [0, 2]], axis=0), axis=-1) * FPS
    return {
        SKATE: statistics[SKATE],
        JERK: statistics[JERK],
        ROOT_SPEED: float(np.percentile(root_speed, OUTLIER_PERCENTILE)),
        ROOT_MEDIAN: float(np.median(root_speed)),
        PLANTED_FOOT: __planted_foot_speed(joints),
    }


def __planted_foot_speed(joints: np.ndarray) -> float:
    foot = joints[:, LEFT_FOOT]
    speed = np.linalg.norm(np.diff(foot[:, [0, 2]], axis=0), axis=-1) * FPS
    height = foot[1:, 1] - joints[:, FOOT_JOINTS, 1].min()
    planted = height < PLANTED_HEIGHT
    return float(np.median(speed[planted])) if planted.any() else 0.0


def __belt_walk(row: dict[str, float]) -> bool:
    return row[ROOT_MEDIAN] < BELT_ROOT_SPEED and row[PLANTED_FOOT] > BELT_FOOT_SPEED


def __group(row: dict[str, float]) -> str:
    return GROUPS[0] if row[ROOT_MEDIAN] >= LOCOMOTION_SPEED else GROUPS[1]


def __grouped(table: dict[str, dict[str, float]], group: str) -> dict[str, dict[str, float]]:
    return {name: row for name, row in table.items() if __group(row) == group}


def __matched_summary(
    reference: dict[str, dict[str, float]], measured: dict[str, dict[str, float]]
) -> dict[str, object]:
    summary: dict[str, object] = {}
    passed = True
    for group in GROUPS:
        source_rows = __grouped(measured, group)
        reference_median = float(np.median(__column(__grouped(reference, group), SKATE)))
        summary[f"{group}_clips"] = len(source_rows)
        if len(source_rows) < MIN_GROUP_CLIPS:
            continue
        ratio = float(np.median(__column(source_rows, SKATE))) / reference_median
        summary[f"{group}_skate_ratio"] = round(ratio, 3)
        passed = passed and ratio <= SKATE_RATIO_LIMIT
    summary["pass_matched_skate"] = passed
    return summary


def __table(paths: list[pathlib.Path]) -> dict[str, dict[str, float]]:
    table = {}
    for path in paths:
        features = np.load(path)
        if not np.isfinite(features).all():
            raise ValueError(f"{path}: non-finite features")
        table[path.stem] = __clip_statistics(features)
    if not table:
        raise FileNotFoundError("no clips to measure")
    return table


def __column(table: dict[str, dict[str, float]], key: str) -> np.ndarray:
    return np.array([row[key] for row in table.values()])


def main() -> None:
    source = sys.argv[1] if len(sys.argv) == 2 else ""
    source_dir = EXTRA / "features20" / source
    if not source_dir.is_dir():
        raise SystemExit(f"usage: physics_check.py <source>; no folder {source_dir}")
    reference_paths = [
        HML3D / "new_joint_vecs" / f"{name}.npy" for name in base_clip_ids(HML3D, "val")
    ]
    reference = __table([path for path in reference_paths if path.is_file()])
    measured = __table(sorted(source_dir.glob("*.npy")))
    skate_limit = np.percentile(__column(reference, SKATE), OUTLIER_PERCENTILE)
    jerk_limit = np.percentile(__column(reference, JERK), OUTLIER_PERCENTILE)
    group_skate_limits = {
        group: np.percentile(__column(__grouped(reference, group), SKATE), OUTLIER_PERCENTILE)
        for group in GROUPS
    }
    clip_rows = []
    for name, row in measured.items():
        outlier = row[SKATE] > skate_limit or row[JERK] > jerk_limit
        group = __group(row)
        matched_outlier = (
            row[SKATE] > group_skate_limits[group]
            or row[JERK] > jerk_limit
            or TREADMILL in name.lower()
            or __belt_walk(row)
        )
        clip_rows.append(
            (
                name,
                round(row[SKATE], 4),
                round(row[JERK], 4),
                round(row[ROOT_SPEED], 3),
                outlier,
                group,
                matched_outlier,
                __belt_walk(row),
            )
        )
    medians = {key: float(np.median(__column(measured, key))) for key in (SKATE, JERK)}
    reference_medians = {key: float(np.median(__column(reference, key))) for key in (SKATE, JERK)}
    root_speed = float(np.percentile(__column(measured, ROOT_SPEED), OUTLIER_PERCENTILE))
    reference_root_speed = float(np.percentile(__column(reference, ROOT_SPEED), OUTLIER_PERCENTILE))
    skate_ratio = medians[SKATE] / reference_medians[SKATE]
    jerk_ratio = medians[JERK] / reference_medians[JERK]
    root_ratio = root_speed / reference_root_speed
    summary = (
        source,
        len(measured),
        round(medians[SKATE], 4),
        round(reference_medians[SKATE], 4),
        round(skate_ratio, 3),
        round(medians[JERK], 4),
        round(reference_medians[JERK], 4),
        round(jerk_ratio, 3),
        round(root_speed, 3),
        round(reference_root_speed, 3),
        round(root_ratio, 3),
        round(sum(row[4] for row in clip_rows) / len(clip_rows), 4),
        skate_ratio <= SKATE_RATIO_LIMIT,
        jerk_ratio <= JERK_RATIO_LIMIT,
        root_ratio <= ROOT_SPEED_RATIO_LIMIT,
    )
    with open(EXTRA / f"physics_{source}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(CLIP_COLUMNS)
        writer.writerows(clip_rows)
    matched = __matched_summary(reference, measured)
    matched["belt_walk_clips"] = sum(row[7] for row in clip_rows)
    matched["matched_outlier_share"] = round(sum(row[6] for row in clip_rows) / len(clip_rows), 4)
    print(dict(zip(SUMMARY_COLUMNS, summary, strict=True)) | matched)


if __name__ == "__main__":
    main()
