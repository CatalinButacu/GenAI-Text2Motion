# .\.venv\Scripts\python.exe scripts/qa/visual_check_render.py
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from text2motion.evaluation.metrics import motion_statistics
from text2motion.motion.representation import FPS, KINEMATIC_CHAINS, recover_from_ric

REPO = pathlib.Path(__file__).resolve().parents[2]
HML3D = REPO / "data" / "HumanML3D_official" / "new_joint_vecs"
EXTRA = REPO / "data" / "extra_motion"
OUT_DIR = REPO / "runs" / "outputs" / "visual_check"
KEY_CSV = REPO / "runs" / "outputs" / "visual_check_key.csv"
JOINT_COUNT = 22
SEED = 2026
PER_GROUP = 10
SKATE_FACTOR = 1.5
LOCOMOTION_SPEED = 0.5
MAX_SECONDS = 8
GRID_STEP_M = 0.5
MARGIN_M = 0.6
CANDIDATE_GROUPS = (("motionx_lp6", ""), ("amass_extra", "GRAB"))


def load_joints(path: pathlib.Path) -> np.ndarray:
    features = torch.from_numpy(np.load(path).astype(np.float32))
    return recover_from_ric(features, JOINT_COUNT).numpy()


def motion_group(joints: np.ndarray) -> str:
    speed = np.linalg.norm(np.diff(joints[:, 0, [0, 2]], axis=0), axis=-1) * FPS
    return "locomotion" if np.median(speed) >= LOCOMOTION_SPEED else "other"


def quality_rows() -> list[dict[str, str]]:
    with open(EXTRA / "quality_clips.csv", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["inlier"] == "True"]


def reference_medians(val_rows: list[dict[str, str]]) -> dict[str, float]:
    values: dict[str, list[float]] = {"locomotion": [], "other": []}
    for row in val_rows:
        joints = load_joints(HML3D / f"{row['name']}.npy")
        values[motion_group(joints)].append(motion_statistics(joints)["foot_skate_m_per_s"])
    return {group: float(np.median(skates)) for group, skates in values.items()}


def candidates(
    rows: list[dict[str, str]], source: str, subset: str, reference: dict[str, float]
) -> list[pathlib.Path]:
    with open(EXTRA / f"physics_{source}.csv", newline="", encoding="utf-8") as handle:
        skate = {row["name"]: row for row in csv.DictReader(handle)}
    picked = []
    for row in rows:
        if (
            row["source"] != source
            or (subset and row["subset"] != subset)
            or row["name"] not in skate
        ):
            continue
        physics = skate[row["name"]]
        if float(physics["foot_skate_m_per_s"]) > SKATE_FACTOR * reference[physics["group"]]:
            picked.append(EXTRA / "features20" / source / f"{row['name']}.npy")
    return picked


def sample(paths: list[pathlib.Path], rng: np.random.Generator) -> list[pathlib.Path]:
    if len(paths) < PER_GROUP:
        raise ValueError(f"only {len(paths)} clips to sample, need {PER_GROUP}")
    return [paths[int(index)] for index in rng.choice(len(paths), PER_GROUP, replace=False)]


def draw(axis, joints: np.ndarray, frame: int, bounds: tuple[float, float, float, float]) -> None:
    x_low, x_high, z_low, z_high = bounds
    axis.clear()
    for x in np.arange(np.floor(x_low), x_high + GRID_STEP_M, GRID_STEP_M):
        axis.plot([x, x], [z_low, z_high], [0, 0], color="#bbbbbb", linewidth=0.6)
    for z in np.arange(np.floor(z_low), z_high + GRID_STEP_M, GRID_STEP_M):
        axis.plot([x_low, x_high], [z, z], [0, 0], color="#bbbbbb", linewidth=0.6)
    pose = joints[frame]
    for chain in KINEMATIC_CHAINS:
        axis.plot(pose[list(chain), 0], pose[list(chain), 2], pose[list(chain), 1], color="#1f3b73")
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(z_low, z_high)
    axis.set_zlim(0, max(x_high - x_low, 2.0))
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=20, azim=-60)
    axis.set_axis_off()


def render(path: pathlib.Path, gif: pathlib.Path) -> None:
    joints = load_joints(path)[: MAX_SECONDS * FPS]
    centre_x, centre_z = joints[:, 0, 0].mean(), joints[:, 0, 2].mean()
    half = max(np.ptp(joints[..., 0]), np.ptp(joints[..., 2]), 1.0) / 2 + MARGIN_M
    bounds = (centre_x - half, centre_x + half, centre_z - half, centre_z + half)
    figure = plt.figure(figsize=(4, 4))
    axis = figure.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=FPS)
    with writer.saving(figure, str(gif), dpi=100):
        for frame in range(len(joints)):
            draw(axis, joints, frame, bounds)
            axis.set_title(gif.stem, fontsize=10)
            writer.grab_frame()
    plt.close(figure)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = quality_rows()
    val_rows = [row for row in rows if row["source"] == "hml3d_val"]
    reference = reference_medians(val_rows)
    clips = [
        ("control", path)
        for path in sample([HML3D / f"{row['name']}.npy" for row in val_rows], rng)
    ]
    for source, subset in CANDIDATE_GROUPS:
        paths = candidates(rows, source, subset, reference)
        clips += [(subset or source, path) for path in sample(paths, rng)]
    order = rng.permutation(len(clips))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = []
    for position, index in enumerate(order, start=1):
        label = f"A{position:02d}"
        group, path = clips[int(index)]
        render(path, OUT_DIR / f"{label}.gif")
        key.append((label, group, path.stem))
        print(label, flush=True)
    with open(KEY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("label", "group", "clip"))
        writer.writerows(key)
    with open(OUT_DIR / "ratings.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("label", "feet_slide_visibly_yes_no"))
        writer.writerows((label, "") for label, _, _ in key)
    print({"reference_skate": reference, "clips": len(key)})


if __name__ == "__main__":
    main()
