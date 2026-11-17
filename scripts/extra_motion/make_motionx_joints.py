# .\.venv\Scripts\python.exe scripts/extra_motion/make_motionx_joints.py animation humman perform kungfu haa500 music idea400
import csv
import io
import pathlib
import re
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO / "data" / "extra_motion" / "motionx_pp"
JOINTS_OUT = REPO / "data" / "extra_motion" / "joints30" / "motionx_official"
INVENTORY = REPO / "data" / "extra_motion" / "inventory_motionx_official.csv"
SOURCE_FPS = 30
BODY_JOINTS = 22
OFFICIAL_JOINTS = 144
CENTIMETRES_PER_METRE = 100.0
SEPARATOR = "__"
MIN_UPRIGHT_SHARE = 0.5
UNSAFE_FILE_CHARACTERS = r'[<>:"/\\|?*]'
INVENTORY_COLUMNS = (
    "name",
    "subset",
    "frames",
    "seconds",
    "upright_share",
    "foot_floor_p05_cm",
    "up_axis_ok",
    "caption",
)


def __captions(subset: str) -> dict[str, str]:
    path = SOURCE / "semantic_label" / f"{subset}.zip"
    with zipfile.ZipFile(path) as archive:
        return {
            pathlib.PurePosixPath(member).stem: archive.read(member).decode("utf-8").strip()
            for member in archive.namelist()
            if member.endswith(".txt")
        }


def __body_joints(official: np.ndarray, name: str) -> np.ndarray:
    if official.ndim != 3 or official.shape[1:] != (OFFICIAL_JOINTS, 3):
        raise ValueError(f"{name}: expected (frames, {OFFICIAL_JOINTS}, 3), got {official.shape}")
    joints = official[:, :BODY_JOINTS].astype(np.float32)
    if not np.isfinite(joints).all():
        raise ValueError(f"{name}: non-finite joints")
    return joints


def __upright_share(joints_m: np.ndarray) -> float:
    head_minus_feet = joints_m[:, 15] - joints_m[:, [10, 11]].mean(axis=1)
    return float((head_minus_feet[:, 1] > 0).mean())


def __foot_floor_p05_cm(joints_m: np.ndarray) -> float:
    foot_height = joints_m[:, [10, 11, 7, 8], 1].min(axis=1)
    return float(np.percentile(foot_height - foot_height.min(), 5) * CENTIMETRES_PER_METRE)


def __subset_rows(subset: str) -> list[tuple]:
    captions = __captions(subset)
    rows = []
    with zipfile.ZipFile(SOURCE / "joint" / f"{subset}.zip") as archive:
        members = sorted(member for member in archive.namelist() if member.endswith(".npy"))
        for member in members:
            stem = pathlib.PurePosixPath(member).stem
            name = re.sub(UNSAFE_FILE_CHARACTERS, "_", f"{subset}{SEPARATOR}{stem}")
            joints = __body_joints(np.load(io.BytesIO(archive.read(member))), name)
            share = __upright_share(joints)
            if share >= MIN_UPRIGHT_SHARE:
                np.save(JOINTS_OUT / f"{name}.npy", joints)
            rows.append(
                (
                    name,
                    subset,
                    joints.shape[0],
                    round(joints.shape[0] / SOURCE_FPS, 2),
                    round(share, 3),
                    round(__foot_floor_p05_cm(joints), 2),
                    share >= MIN_UPRIGHT_SHARE,
                    captions.get(stem, ""),
                )
            )
    return rows


def __previous_rows() -> list[list[str]]:
    if not INVENTORY.is_file():
        return []
    with open(INVENTORY, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))[1:]


def main() -> None:
    subsets = sys.argv[1:]
    if not subsets:
        raise SystemExit("usage: make_motionx_joints.py <subset> [<subset> ...]")
    JOINTS_OUT.mkdir(parents=True, exist_ok=True)
    kept = [row for row in __previous_rows() if row[1] not in subsets]
    rows = []
    for subset in subsets:
        subset_rows = __subset_rows(subset)
        no_caption = sum(1 for row in subset_rows if not row[7])
        bad_axis = sum(1 for row in subset_rows if not row[6])
        hours = sum(row[3] for row in subset_rows) / 3600
        print(
            f"{subset}: clips {len(subset_rows)}  hours {hours:.2f}  not upright {bad_axis}  no caption {no_caption}"
        )
        rows += subset_rows
    with open(INVENTORY, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(INVENTORY_COLUMNS)
        writer.writerows(kept)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
