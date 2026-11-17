# .\.venv\Scripts\python.exe scripts/extra_motion/make_style100_joints.py
import csv
import io
import pathlib
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE_ZIP = REPO / "data" / "extra_motion" / "style100" / "100STYLE.zip"
JOINTS_OUT = REPO / "data" / "extra_motion" / "joints60" / "style100"
INVENTORY = REPO / "data" / "extra_motion" / "inventory_style100.csv"
SOURCE_FPS = 60
CENTIMETRES_PER_METRE = 100.0
CUTS_NAME = "100STYLE/Frame_Cuts.csv"
BVH_SUFFIX = ".bvh"
HUMANML3D_JOINTS = (
    "Hips",
    "LeftHip",
    "RightHip",
    "Chest",
    "LeftKnee",
    "RightKnee",
    "Chest2",
    "LeftAnkle",
    "RightAnkle",
    "Chest4",
    "LeftToe",
    "RightToe",
    "Neck",
    "LeftCollar",
    "RightCollar",
    "Head",
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftWrist",
    "RightWrist",
)
INVENTORY_COLUMNS = ("name", "frames_total", "frames_kept", "seconds_kept", "up_axis_ok")
IDENTITY = np.eye(3)


class _Skeleton:
    def __init__(self, names, parents, offsets, channels):
        self.names = names
        self.parents = parents
        self.offsets = offsets
        self.channels = channels


def __parse_hierarchy(header: str) -> _Skeleton:
    names, parents, offsets, channels = [], [], [], []
    stack: list[int] = []
    tokens = header.replace("{", " { ").replace("}", " } ").split()
    index = 0
    pending_end_site = False
    while index < len(tokens):
        token = tokens[index]
        if token in ("ROOT", "JOINT"):
            names.append(tokens[index + 1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(None)
            channels.append(None)
            stack.append(len(names) - 1)
            index += 2
        elif token == "End":
            pending_end_site = True
            index += 2
        elif token == "OFFSET":
            if not pending_end_site:
                offsets[stack[-1]] = [float(value) for value in tokens[index + 1 : index + 4]]
            index += 4
        elif token == "CHANNELS":
            count = int(tokens[index + 1])
            channels[stack[-1]] = tokens[index + 2 : index + 2 + count]
            index += 2 + count
        elif token == "}":
            if pending_end_site:
                pending_end_site = False
            else:
                stack.pop()
            index += 1
        else:
            index += 1
    if any(value is None for value in offsets) or any(value is None for value in channels):
        raise ValueError("a joint has no OFFSET or CHANNELS")
    return _Skeleton(names, parents, np.asarray(offsets, dtype=np.float64), channels)


def __axis_rotation(axis: str, degrees: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.tile(IDENTITY, (angle.shape[0], 1, 1))
    first, second = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    matrix[:, first, first] = cos
    matrix[:, first, second] = -sin
    matrix[:, second, first] = sin
    matrix[:, second, second] = cos
    return matrix


def __joint_rotation(channel_names: list[str], values: np.ndarray) -> np.ndarray:
    rotation = np.tile(IDENTITY, (values.shape[0], 1, 1))
    for name, column in zip(channel_names, values.T, strict=True):
        if name.endswith("rotation"):
            rotation = rotation @ __axis_rotation(name[0], column)
    return rotation


def __forward_kinematics(skeleton: _Skeleton, motion: np.ndarray) -> np.ndarray:
    frames = motion.shape[0]
    positions = np.zeros((len(skeleton.names), frames, 3))
    rotations: dict[int, np.ndarray] = {}
    column = 0
    for joint, channel_names in enumerate(skeleton.channels):
        values = motion[:, column : column + len(channel_names)]
        column += len(channel_names)
        local = __joint_rotation(channel_names, values)
        parent = skeleton.parents[joint]
        if parent < 0:
            shift = np.stack(
                [values[:, channel_names.index(f"{axis}position")] for axis in "XYZ"], axis=1
            )
            positions[joint] = shift + skeleton.offsets[joint]
            rotations[joint] = local
            continue
        positions[joint] = positions[parent] + np.einsum(
            "tij,j->ti", rotations[parent], skeleton.offsets[joint]
        )
        rotations[joint] = rotations[parent] @ local
    if column != motion.shape[1]:
        raise ValueError(f"channels use {column} columns, the data has {motion.shape[1]}")
    return positions


def __read_bvh(raw: bytes) -> tuple[_Skeleton, np.ndarray]:
    text = raw.decode("utf-8", errors="ignore")
    split = text.index("MOTION")
    skeleton = __parse_hierarchy(text[:split])
    lines = text[split:].split("\n", 3)
    frames = int(lines[1].split(":")[1])
    frame_time = float(lines[2].split(":")[1])
    if abs(1.0 / frame_time - SOURCE_FPS) > 0.5:
        raise ValueError(f"frame time {frame_time} is not {SOURCE_FPS} fps")
    motion = np.array(lines[3].split(), dtype=np.float64).reshape(frames, -1)
    return skeleton, motion


def __read_cuts(archive: zipfile.ZipFile) -> dict[str, tuple[int, int]]:
    cuts = {}
    reader = csv.DictReader(io.StringIO(archive.read(CUTS_NAME).decode("utf-8", errors="ignore")))
    for row in reader:
        for key in row:
            if not key.endswith("_START"):
                continue
            kind = key[: -len("_START")]
            start, stop = row[f"{kind}_START"], row[f"{kind}_STOP"]
            if start != "N/A":
                cuts[f"{row['STYLE_NAME']}_{kind}"] = (int(start), int(stop))
    return cuts


def __up_axis_ok(joints_m: np.ndarray) -> bool:
    head_minus_feet = joints_m[:, 15] - joints_m[:, [10, 11]].mean(axis=1)
    return bool(np.abs(head_minus_feet.mean(axis=0)).argmax() == 1)


def main() -> None:
    JOINTS_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        cuts = __read_cuts(archive)
        members = sorted(
            info.filename
            for info in archive.infolist()
            if info.filename.endswith(BVH_SUFFIX) and "__MACOSX" not in info.filename
        )
        for member in members:
            stem = pathlib.PurePosixPath(member).stem
            if stem not in cuts:
                raise KeyError(f"{stem}: no frame cut in Frame_Cuts.csv")
            skeleton, motion = __read_bvh(archive.read(member))
            order = [skeleton.names.index(name) for name in HUMANML3D_JOINTS]
            joints = __forward_kinematics(skeleton, motion)[order].transpose(1, 0, 2)
            start, stop = cuts[stem]
            kept = joints[start:stop] / CENTIMETRES_PER_METRE
            if kept.shape[0] < 2 or not np.isfinite(kept).all():
                raise ValueError(f"{stem}: cut {start}:{stop} leaves {kept.shape[0]} frames")
            np.save(JOINTS_OUT / f"{stem}.npy", kept.astype(np.float32))
            rows.append(
                (
                    stem,
                    motion.shape[0],
                    kept.shape[0],
                    round(kept.shape[0] / SOURCE_FPS, 2),
                    __up_axis_ok(kept),
                )
            )
    with open(INVENTORY, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(INVENTORY_COLUMNS)
        writer.writerows(rows)
    hours = sum(row[2] for row in rows) / SOURCE_FPS / 3600
    bad = sum(1 for row in rows if not row[4])
    print(f"clips {len(rows)}  hours kept {hours:.2f}  wrong up axis {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
