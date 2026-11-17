# .\.venv\Scripts\python.exe scripts/extra_motion/make_amass_joints.py control
import csv
import pathlib
import re
import sys
from types import MappingProxyType

import numpy as np

from text2motion.motion.amass import AmassPoseExtractor, head_trim_frames
from text2motion.motion.representation import FPS, JOINTS

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = pathlib.Path(r"D:\Facultate\dissertation\data\amass")
HML3D = REPO / "data" / "HumanML3D_official"
EXTRA = REPO / "data" / "extra_motion"
SMPLX_MODELS = REPO / "data" / "smplx_models"
CONTROL_CLIPS = 200
SEED = 2026
MIN_UPRIGHT_SHARE = 0.5
SEPARATOR = "__"
UNSAFE_FILE_CHARACTERS = r'[<>:"/\\|?* ]'
DATASET_ALIASES = MappingProxyType(
    {
        "biomotionlab_ntroje": "bmlrub",
        "dfaust_67": "dfaust",
        "mpi_hdm05": "hdm05",
        "mpi_limits": "poseprior",
        "mpi_mosh": "mosh",
        "ssm_synced": "ssm",
        "transitions_mocap": "transitions",
        "eyesjapandataset": "eyes_japan_dataset",
    }
)
MODES = ("control", "extra")
INVENTORY_COLUMNS = ("name", "raw_path", "frames", "seconds", "upright_share", "up_axis_ok")


def __key(path: str) -> str:
    parts = [
        part
        for part in re.split(r"[\\/]", path.strip().lstrip("./"))
        if part and part != "pose_data"
    ]
    dataset = DATASET_ALIASES.get(parts[0].lower(), parts[0].lower())
    stem = re.sub(r"_(poses|stageii)$", "", pathlib.Path(parts[-1]).stem.lower())
    subject = parts[-2].lower() if len(parts) > 2 else ""
    return f"{dataset}/{subject.replace(' ', '_')}/{stem.replace(' ', '_')}"


def __hml3d_splits() -> dict[str, set[str]]:
    split_of = {}
    for split in ("train", "val", "test"):
        for line in (HML3D / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                split_of[line.strip().removeprefix("M")] = split
    splits: dict[str, set[str]] = {}
    with open(HML3D / "index.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "humanact12" in row["source_path"].lower():
                continue
            name = row["new_name"].removesuffix(".npy")
            splits.setdefault(__key(row["source_path"]), set()).add(split_of.get(name, "none"))
    return splits


def __selected(mode: str) -> list[pathlib.Path]:
    files = sorted(path for path in RAW.rglob("*_stageii.npz") if "MOYO" not in path.parts)
    if not files:
        raise FileNotFoundError(f"no raw AMASS files under {RAW}")
    splits = __hml3d_splits()
    keys = {path: __key(path.relative_to(RAW).as_posix()) for path in files}
    if mode == "extra":
        return [path for path in files if keys[path] not in splits]
    train_only = [path for path in files if splits.get(keys[path]) == {"train"}]
    order = np.random.default_rng(SEED).permutation(len(train_only))[:CONTROL_CLIPS]
    return sorted(train_only[int(index)] for index in order)


def __upright_share(joints: np.ndarray) -> float:
    head_minus_feet = joints[:, 15] - joints[:, [10, 11]].mean(axis=1)
    return float((head_minus_feet[:, 1] > 0).mean())


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) in (2, 3) else ""
    if mode not in MODES:
        raise SystemExit(f"usage: make_amass_joints.py <{'|'.join(MODES)}> [fps]")
    fps = int(sys.argv[2]) if len(sys.argv) == 3 else FPS
    out_dir = EXTRA / f"joints{fps}" / f"amass_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = AmassPoseExtractor(SMPLX_MODELS, joint_count=JOINTS, fps=fps)
    rows = []
    for path in __selected(mode):
        relative = path.relative_to(RAW).as_posix()
        joints = extractor.extract(path)
        if joints is None:
            continue
        joints = joints[head_trim_frames(relative, fps) :].copy()
        joints[..., 0] *= -1
        name = re.sub(
            UNSAFE_FILE_CHARACTERS, "_", relative.removesuffix(".npz").replace("/", SEPARATOR)
        )
        share = __upright_share(joints)
        if share >= MIN_UPRIGHT_SHARE:
            np.save(out_dir / f"{name}.npy", joints)
        seconds = round(joints.shape[0] / fps, 2)
        rows.append(
            (name, relative, joints.shape[0], seconds, round(share, 3), share >= MIN_UPRIGHT_SHARE)
        )
    suffix = "" if fps == FPS else f"_{fps}fps"
    inventory = EXTRA / f"inventory_amass_{mode}{suffix}.csv"
    with open(inventory, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(INVENTORY_COLUMNS)
        writer.writerows(rows)
    hours = sum(row[3] for row in rows) / 3600
    not_upright = sum(not row[5] for row in rows)
    print(f"{mode}: files {len(rows)}  hours {hours:.2f}  not upright {not_upright}")


if __name__ == "__main__":
    main()
