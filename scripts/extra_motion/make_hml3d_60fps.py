# .\.venv\Scripts\python.exe scripts/extra_motion/make_hml3d_60fps.py [features]
import csv
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from text2motion.motion.amass import AmassPoseExtractor, resolve_pose_path
from text2motion.motion.contracts import PreparationRequest, PreparationStage
from text2motion.motion.preparation import (
    INDEX_MANIFEST,
    Hml3dPreparationLayout,
    build_indexed_joint_clips,
    index_frame_scale,
    run_hml3d_preparation,
)
from text2motion.motion.representation import FPS, JOINTS, recover_from_ric

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = pathlib.Path(r"D:\Facultate\dissertation\data\amass")
OFFICIAL = REPO / "data" / "HumanML3D_official"
OUT = REPO / "data" / "HumanML3D_60fps"
CLOCKED_POSES = OUT / "pose_data_clocked"
CLOCK_CSV = REPO / "doc" / "notes" / "amass_clock_check.csv"
SMPLX_MODELS = REPO / "data" / "smplx_models"
TARGET_FPS = 60
USABLE_CLOCK_STATUS = frozenset({"ok", "folder_rate"})
LABEL_FPS = 120
SPLIT_FILES = ("train.txt", "val.txt", "test.txt")


def __spline_to_target(joints: np.ndarray, source_fps: float) -> np.ndarray:
    times = np.arange(joints.shape[0]) / source_fps
    targets = np.arange(int(np.floor(times[-1] * TARGET_FPS)) + 1) / TARGET_FPS
    curve = CubicSpline(times, joints.astype(np.float64), axis=0)
    return curve(targets).astype(np.float32)


def __clocked_files(layout: Hml3dPreparationLayout) -> list[tuple[pathlib.Path, int]]:
    with open(CLOCK_CSV, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] in USABLE_CLOCK_STATUS]
    files = []
    for row in rows:
        pose_path = resolve_pose_path(row["source_path"], layout.pose_data)
        if pose_path is None:
            raise FileNotFoundError(f"clock table names {row['source_path']} but no pose file")
        files.append((pose_path, int(row["stored_fps"])))
    return files


def __write_clocked_poses(layout: Hml3dPreparationLayout) -> dict[int, int]:
    extractor = AmassPoseExtractor(SMPLX_MODELS, joint_count=JOINTS, fps=LABEL_FPS)
    counts: dict[int, int] = {}
    for pose_path, stored_fps in tqdm(__clocked_files(layout), desc="clocked poses"):
        relative = pose_path.relative_to(layout.pose_data)
        target = CLOCKED_POSES / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        counts[stored_fps] = counts.get(stored_fps, 0) + 1
        if target.is_file():
            continue
        if stored_fps == LABEL_FPS:
            os.link(pose_path, target)
            continue
        joints = extractor.extract(RAW / relative.with_suffix(".npz"))
        if joints is None:
            raise ValueError(f"{relative} has no motion at the stored rate")
        np.save(target, __spline_to_target(joints, stored_fps))
    return counts


def __official_joints_at_target(name: str) -> np.ndarray:
    features = np.load(OFFICIAL / "new_joint_vecs" / f"{name}.npy").astype(np.float32)
    joints = recover_from_ric(torch.from_numpy(features), JOINTS).numpy().astype(np.float64)
    times = np.arange(joints.shape[0]) / FPS
    targets = np.arange((joints.shape[0] - 1) * index_frame_scale(TARGET_FPS) + 1) / TARGET_FPS
    return CubicSpline(times, joints, axis=0)(targets).astype(np.float32)


def __fill_from_official(layout: Hml3dPreparationLayout) -> int:
    manifest = json.loads((OUT / INDEX_MANIFEST).read_text(encoding="utf-8"))
    split_names = {
        line.strip().removeprefix("M")
        for split in SPLIT_FILES
        for line in (OFFICIAL / split).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    missing = [str(name).removesuffix(".npy") for name in manifest["missing_names"]]
    names = [name for name in missing if name in split_names]
    print("missing clips in no split, skipped:", sorted(set(missing) - split_names), flush=True)
    absent = [name for name in names if not (OFFICIAL / "new_joint_vecs" / f"{name}.npy").is_file()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} clips have no official features; first: {absent[:3]}"
        )
    one_frame = [
        name for name in names if np.load(OFFICIAL / "new_joint_vecs" / f"{name}.npy").shape[0] < 2
    ]
    print("official clips with 1 frame, skipped:", one_frame, flush=True)
    filled = [name for name in names if name not in one_frame]
    for name in tqdm(filled, desc="fill from official"):
        for prefix in ("", "M"):
            np.save(
                layout.joints / f"{prefix}{name}.npy",
                __official_joints_at_target(f"{prefix}{name}"),
            )
    return len(filled)


def __build_joint_clips(request: PreparationRequest, layout: Hml3dPreparationLayout) -> None:
    run_hml3d_preparation(request, PreparationStage.AMASS)
    print("clocked pose files by stored fps:", __write_clocked_poses(layout), flush=True)
    shutil.rmtree(layout.joints)
    layout.mkdirs()
    build_indexed_joint_clips(request, layout, CLOCKED_POSES)
    print("clips filled from official features:", __fill_from_official(layout), flush=True)


def main() -> None:
    request = PreparationRequest(
        out_dir=OUT,
        amass_dir=RAW,
        smplx_models=SMPLX_MODELS,
        index_csv=OFFICIAL / "index.csv",
        fps=TARGET_FPS,
    )
    layout = Hml3dPreparationLayout(OUT)
    layout.mkdirs()
    for split in SPLIT_FILES:
        shutil.copyfile(OFFICIAL / split, OUT / split)
    if sys.argv[1:] != ["features"]:
        __build_joint_clips(request, layout)
    run_hml3d_preparation(request, PreparationStage.FEATURE)
    run_hml3d_preparation(request, PreparationStage.STATS)
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
