# .\.venv\Scripts\python.exe scripts/qa/data_gallery_render.py
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from text2motion.motion.representation import KINEMATIC_CHAINS

REPO = pathlib.Path(__file__).resolve().parents[2]
EXTRA = REPO / "data" / "extra_motion"
OUT_DIR = REPO / "artifacts" / "outputs" / "data_gallery_60fps"
SHEET = OUT_DIR / "gallery.csv"
FPS = 60
SEED = 2026
PER_GROUP = 5
MAX_SECONDS = 8
MAX_VIOLATION_SHARE = 0.5
GRID_STEP_M = 0.5
MARGIN_M = 0.6
PIXELS = 480
GROUPS = (
    ("amass_extra", "amass_extra", "GRAB"),
    ("amass_extra", "amass_extra", "BMLrub"),
    ("amass_extra", "amass_extra", "HUMAN4D"),
    ("motionx_official_lp6", "motionx_official", "idea400"),
    ("motionx_official_lp6", "motionx_official", "music"),
    ("motionx_official_lp6", "motionx_official", "perform"),
    ("motionx_official_lp6", "motionx_official", "kungfu"),
    ("motionx_official_lp6", "motionx_official", "haa500"),
    ("motionx_official_lp6", "motionx_official", "humman"),
    ("motionx_official_lp6", "motionx_official", "animation"),
)
SHEET_COLUMNS = ("label", "source", "type", "clip", "seconds", "caption", "owner_note")


def __kept_names() -> set[tuple[str, str]]:
    with open(EXTRA / "quality_clips.csv", newline="", encoding="utf-8") as handle:
        inliers = {
            (row["source"], row["name"])
            for row in csv.DictReader(handle)
            if row["inlier"] == "True"
        }
    with open(EXTRA / "newton_clips.csv", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (row["source"], row["name"])
            for row in rows
            if float(row["violation_share"]) < MAX_VIOLATION_SHARE
            and (row["source"], row["name"]) in inliers
        }


def __captions() -> dict[str, str]:
    with open(EXTRA / "inventory_motionx_official.csv", newline="", encoding="utf-8") as handle:
        return {row["name"]: row["caption"].split(". ")[0] for row in csv.DictReader(handle)}


def __picked(
    kept: set[tuple[str, str]], rng: np.random.Generator
) -> list[tuple[str, str, pathlib.Path]]:
    picked = []
    for score_source, joints_source, subset in GROUPS:
        folder = EXTRA / f"joints{FPS}" / joints_source
        names = sorted(
            name
            for source, name in kept
            if source == score_source
            and name.split("__")[0] == subset
            and (folder / f"{name}.npy").is_file()
        )
        if len(names) < PER_GROUP:
            raise ValueError(f"{subset}: only {len(names)} kept clips at {FPS} fps")
        for index in rng.choice(len(names), PER_GROUP, replace=False):
            picked.append((joints_source, subset, folder / f"{names[int(index)]}.npy"))
    return picked


def __draw(axis, joints: np.ndarray, frame: int, bounds: tuple[float, float, float, float]) -> None:
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


def __render(path: pathlib.Path, video: pathlib.Path, title: str) -> float:
    joints = np.load(path)[: MAX_SECONDS * FPS].astype(np.float64)
    joints[..., 1] -= joints[..., 1].min()
    centre_x, centre_z = joints[:, 0, 0].mean(), joints[:, 0, 2].mean()
    half = max(np.ptp(joints[..., 0]), np.ptp(joints[..., 2]), 1.0) / 2 + MARGIN_M
    bounds = (centre_x - half, centre_x + half, centre_z - half, centre_z + half)
    figure = plt.figure(figsize=(PIXELS / 100, PIXELS / 100), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (PIXELS, PIXELS))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {video}")
    for frame in range(len(joints)):
        __draw(axis, joints, frame, bounds)
        axis.set_title(title, fontsize=10)
        canvas = figure.canvas
        if not isinstance(canvas, FigureCanvasAgg):
            raise TypeError(f"expected the Agg canvas, got {type(canvas).__name__}")
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
    writer.release()
    plt.close(figure)
    return len(joints) / FPS


def main() -> None:
    rng = np.random.default_rng(SEED)
    picked = __picked(__kept_names(), rng)
    captions = __captions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, index in enumerate(rng.permutation(len(picked)), start=1):
        source, subset, path = picked[int(index)]
        label = f"G{position:02d}"
        seconds = __render(path, OUT_DIR / f"{label}.mp4", f"{label} {subset}")
        rows.append(
            (label, source, subset, path.stem, round(seconds, 2), captions.get(path.stem, ""), "")
        )
        print(label, subset, flush=True)
    with open(SHEET, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SHEET_COLUMNS)
        writer.writerows(rows)
    print({"videos": len(rows), "fps": FPS, "folder": str(OUT_DIR)})


if __name__ == "__main__":
    main()
