# .\.venv\Scripts\python.exe scripts/extra_motion/newton_check.py
import csv
import pathlib

import numpy as np
import torch

from text2motion.motion.representation import recover_from_ric
from text2motion.motion.storage import base_clip_ids

REPO = pathlib.Path(__file__).resolve().parents[2]
HML3D = REPO / "data" / "HumanML3D_official"
EXTRA = REPO / "data" / "extra_motion"
CLIP_CSV = EXTRA / "newton_clips.csv"
SUMMARY_CSV = REPO / "doc" / "notes" / "newton_check.csv"
QUALITY_CSV = EXTRA / "quality_clips.csv"
FPS = 20
JOINT_COUNT = 22
GRAVITY_M_PER_S2 = 9.81
HOVER_FRACTION_OF_G = 0.5
QUASI_STATIC_M_PER_S2 = 1.0
SUPPORT_MARGIN_M = 0.10
MIN_FLIGHT_FRAMES = 3
MIN_UNBALANCED_FRAMES = 5
LIMIT_PERCENTILE = 99
MIN_FRAMES = 40
SEGMENTS = (
    (12, 15, 6.94),
    (0, 12, 43.46),
    (16, 18, 2.71),
    (17, 19, 2.71),
    (18, 20, 1.62),
    (19, 21, 1.62),
    (20, 20, 0.61),
    (21, 21, 0.61),
    (1, 4, 14.16),
    (2, 5, 14.16),
    (4, 7, 4.33),
    (5, 8, 4.33),
    (7, 10, 1.37),
    (8, 11, 1.37),
)
CONTACT_HEIGHT_M = (
    0.15,
    0.15,
    0.15,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.05,
    0.05,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.08,
    0.08,
)
FEET = ((7, 10), (8, 11))
EXTRA_SOURCES = ("amass_extra", "motionx_lp6", "motionx_official_lp6")
CLIP_COLUMNS = (
    "source",
    "subset",
    "name",
    "seconds",
    "hover_share",
    "unbalanced_share",
    "violation_share",
    "rejected",
)
SUMMARY_COLUMNS = (
    "source",
    "subset",
    "clips",
    "hours",
    "rejected_share",
    "hours_kept_newton",
    "hours_kept_newton_and_t5g",
)


def __centre_of_mass(joints: np.ndarray) -> np.ndarray:
    masses = np.array([mass for _, _, mass in SEGMENTS])
    centres = np.stack([(joints[:, a] + joints[:, b]) / 2 for a, b, _ in SEGMENTS], axis=1)
    return np.einsum("s,fsc->fc", masses, centres) / masses.sum()


def __long_runs(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    runs = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    return [(int(start), int(end)) for start, end in runs if end - start >= min_frames]


def __hover_frames(acceleration: np.ndarray, contact: np.ndarray) -> np.ndarray:
    hover = np.zeros(len(contact), dtype=bool)
    gravity = np.array([0.0, -GRAVITY_M_PER_S2, 0.0])
    for start, end in __long_runs(~contact, MIN_FLIGHT_FRAMES):
        departure = np.linalg.norm(acceleration[start:end] - gravity, axis=1).mean()
        hover[start:end] = departure >= HOVER_FRACTION_OF_G * GRAVITY_M_PER_S2
    return hover


def __whole_feet(touching: np.ndarray) -> np.ndarray:
    support = touching.copy()
    for ankle, toe in FEET:
        foot_down = touching[:, ankle] | touching[:, toe]
        support[:, ankle] = foot_down
        support[:, toe] = foot_down
    return support


def __unbalanced_frames(
    joints: np.ndarray, com: np.ndarray, acceleration: np.ndarray, touching: np.ndarray
) -> np.ndarray:
    support = __whole_feet(touching)
    plane = joints[..., [0, 2]]
    low = np.where(support[..., None], plane, np.inf).min(axis=1) - SUPPORT_MARGIN_M
    high = np.where(support[..., None], plane, -np.inf).max(axis=1) + SUPPORT_MARGIN_M
    centre = com[:, [0, 2]]
    outside = ((centre < low) | (centre > high)).any(axis=1)
    still = np.linalg.norm(acceleration, axis=1) < QUASI_STATIC_M_PER_S2
    candidate = outside & still & touching.any(axis=1)
    unbalanced = np.zeros(len(candidate), dtype=bool)
    for start, end in __long_runs(candidate, MIN_UNBALANCED_FRAMES):
        unbalanced[start:end] = True
    return unbalanced


def __violations(joints: np.ndarray) -> tuple[float, float]:
    heights = joints[..., 1] - joints[..., 1].min()
    touching = heights < np.array(CONTACT_HEIGHT_M)
    com = __centre_of_mass(joints)
    acceleration = np.gradient(np.gradient(com, axis=0), axis=0) * FPS * FPS
    hover = __hover_frames(acceleration, touching.any(axis=1))
    unbalanced = __unbalanced_frames(joints, com, acceleration, touching)
    return float(hover.mean()), float(unbalanced.mean())


def __clip_row(source: str, path: pathlib.Path) -> list | None:
    features = np.load(path).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: non-finite features")
    if features.shape[0] < MIN_FRAMES:
        return None
    joints = recover_from_ric(torch.from_numpy(features), JOINT_COUNT).numpy()
    hover, unbalanced = __violations(joints)
    subset = path.stem.split("__")[0] if "__" in path.stem else source
    seconds = features.shape[0] / FPS
    return [
        source,
        subset,
        path.stem,
        seconds,
        round(hover, 4),
        round(unbalanced, 4),
        round(hover + unbalanced, 4),
    ]


def __measure(source: str, paths: list[pathlib.Path]) -> list[list]:
    rows = [row for row in (__clip_row(source, path) for path in paths) if row is not None]
    print(
        source,
        "clips",
        len(rows),
        "shorter than",
        MIN_FRAMES,
        "frames",
        len(paths) - len(rows),
        flush=True,
    )
    return rows


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
        kept = np.array([not row[7] for row in members])
        both = np.array([not row[7] and (row[0], row[2]) in inliers for row in members])
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


def __hml3d_paths(split: str) -> list[pathlib.Path]:
    paths = [HML3D / "new_joint_vecs" / f"{name}.npy" for name in base_clip_ids(HML3D, split)]
    return [path for path in paths if path.is_file()]


def main() -> None:
    rows = __measure("hml3d_train", __hml3d_paths("train"))
    limit = float(np.percentile([row[6] for row in rows], LIMIT_PERCENTILE))
    rows += __measure("hml3d_val", __hml3d_paths("val"))
    for source in EXTRA_SOURCES:
        rows += __measure(source, sorted((EXTRA / "features20" / source).glob("*.npy")))
    for row in rows:
        row.append(row[6] > limit)
    with open(CLIP_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CLIP_COLUMNS)
        writer.writerows(rows)
    table = __summary(rows, __t5g_inliers())
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(table)
    print("limit_violation_share", round(limit, 4))
    for line in table:
        print(",".join(str(value) for value in line))


if __name__ == "__main__":
    main()
