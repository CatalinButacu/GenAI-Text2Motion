# .\.venv\Scripts\python.exe scripts/extra_motion/glide_check.py
import csv
import pathlib

import numpy as np
import torch

from text2motion.motion.representation import recover_from_ric
from text2motion.motion.storage import base_clip_ids

REPO = pathlib.Path(__file__).resolve().parents[2]
HML3D = REPO / "data" / "HumanML3D_official"
EXTRA = REPO / "data" / "extra_motion"
CLIP_CSV = EXTRA / "glide_clips.csv"
SUMMARY_CSV = REPO / "doc" / "notes" / "glide_check.csv"
QUALITY_CSV = EXTRA / "quality_clips.csv"
FPS = 20
JOINT_COUNT = 22
TOES = (10, 11)
CONTACT_HEIGHT_M = 0.05
SWING_SPEED_M_PER_S = 0.5
LIMIT_PERCENTILE = 99
MIN_FRAMES = 40
EXTRA_SOURCES = ("amass_extra", "motionx_lp6")
CLIP_COLUMNS = ("source", "subset", "name", "seconds", "max_glide_m", "rejected")
SUMMARY_COLUMNS = (
    "source",
    "subset",
    "clips",
    "hours",
    "rejected_share",
    "hours_kept_glide",
    "hours_kept_glide_and_t5g",
)


def __phase_glides(toe: np.ndarray) -> list[float]:
    step = np.linalg.norm(np.diff(toe[:, [0, 2]], axis=0), axis=-1) * FPS
    speed = np.concatenate((step, step[-1:]))
    contact = (toe[:, 1] < CONTACT_HEIGHT_M) & (speed < SWING_SPEED_M_PER_S)
    edges = np.diff(np.concatenate(([0], contact.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    glides = []
    for start, end in zip(starts, ends, strict=True):
        offset = toe[start:end, [0, 2]] - toe[start, [0, 2]]
        glides.append(float(np.linalg.norm(offset, axis=-1).max()))
    return glides


def __clip_glides(path: pathlib.Path) -> tuple[list[float], float] | None:
    features = np.load(path).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: non-finite features")
    if features.shape[0] < MIN_FRAMES:
        return None
    joints = recover_from_ric(torch.from_numpy(features), JOINT_COUNT).numpy()
    glides = [glide for toe in TOES for glide in __phase_glides(joints[:, toe])]
    return glides, features.shape[0] / FPS


def __measure(source: str, paths: list[pathlib.Path]) -> tuple[list[list], list[float]]:
    rows: list[list] = []
    pooled: list[float] = []
    short = 0
    for path in paths:
        measured = __clip_glides(path)
        if measured is None:
            short += 1
            continue
        glides, seconds = measured
        pooled += glides
        subset = path.stem.split("__")[0] if "__" in path.stem else source
        rows.append([source, subset, path.stem, seconds, max(glides, default=0.0)])
    print(source, "clips", len(rows), "shorter than", MIN_FRAMES, "frames", short, flush=True)
    return rows, pooled


def __t5g_inliers() -> set[tuple[str, str]]:
    with open(QUALITY_CSV, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {(row["source"], row["name"]) for row in rows if row["inlier"] == "True"}


def __summary(rows: list[list], inliers: set[tuple[str, str]]) -> list[tuple]:
    groups: dict[tuple[str, str], list[list]] = {}
    for row in rows:
        groups.setdefault((row[0], "all"), []).append(row)
        if row[1] != row[0]:
            groups.setdefault((row[0], row[1]), []).append(row)
    table = []
    for (source, subset), members in sorted(groups.items()):
        seconds = np.array([row[3] for row in members])
        kept = np.array([not row[5] for row in members])
        both = np.array([not row[5] and (row[0], row[2]) in inliers for row in members])
        table.append(
            (
                source,
                subset,
                len(members),
                round(float(seconds.sum()) / 3600, 2),
                round(1 - float(kept.mean()), 3),
                round(float(seconds[kept].sum()) / 3600, 2),
                round(float(seconds[both].sum()) / 3600, 2),
            )
        )
    return table


def main() -> None:
    train_paths = [
        HML3D / "new_joint_vecs" / f"{name}.npy" for name in base_clip_ids(HML3D, "train")
    ]
    val_paths = [HML3D / "new_joint_vecs" / f"{name}.npy" for name in base_clip_ids(HML3D, "val")]
    rows, train_glides = __measure("hml3d_train", [path for path in train_paths if path.is_file()])
    limit = float(np.percentile(train_glides, LIMIT_PERCENTILE))
    rows += __measure("hml3d_val", [path for path in val_paths if path.is_file()])[0]
    for source in EXTRA_SOURCES:
        rows += __measure(source, sorted((EXTRA / "features20" / source).glob("*.npy")))[0]
    for row in rows:
        row.append(row[4] > limit)
    with open(CLIP_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CLIP_COLUMNS)
        writer.writerows(rows)
    table = __summary(rows, __t5g_inliers())
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(table)
    print("limit_m", round(limit, 4), "train phases", len(train_glides))
    for line in table:
        print(",".join(str(value) for value in line))


if __name__ == "__main__":
    main()
