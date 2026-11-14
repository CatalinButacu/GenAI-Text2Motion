# .\.venv\Scripts\python.exe scripts/extra_motion/make_humman_joints.py
import csv
import json
import pathlib
import sys

import numpy as np
import smplx
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO / "data" / "extra_motion" / "humman_mogen" / "data"
JOINTS_OUT = REPO / "data" / "extra_motion" / "joints30" / "humman"
INVENTORY = REPO / "data" / "extra_motion" / "inventory_humman.csv"
SMPLX_MODELS = REPO / "data" / "smplx_models"
SOURCE_FPS = 30
SAMPLE_CLIPS = 200
SEED = 2026
BODY_JOINTS = 22
CHUNK_FRAMES = 256
CENTIMETRES_PER_METRE = 100.0
JUMP_WORD = "jump"
INVENTORY_COLUMNS = (
    "name",
    "frames",
    "seconds",
    "has_jump",
    "foot_floor_p05_cm",
    "up_axis_ok",
)


def __captions(name: str) -> list[str]:
    with open(SOURCE / "annotations" / f"{name}.json", encoding="utf-8") as handle:
        return [stage["all"] for stage in json.load(handle)]


def __has_jump(name: str) -> bool:
    return any(JUMP_WORD in caption.lower() for caption in __captions(name))


def __selected_names() -> list[str]:
    names = sorted(
        path.stem
        for path in (SOURCE / "smpl_on_ground").glob("*.npz")
        if path.stem.split("_")[1].startswith("a")
    )
    if not names:
        raise FileNotFoundError(f"no unmirrored clips under {SOURCE}")
    rng = np.random.default_rng(SEED)
    sample = set(rng.choice(names, SAMPLE_CLIPS, replace=False).tolist())
    jumps = {name for name in names if __has_jump(name)}
    return sorted(sample | jumps)


def __create_model():
    return smplx.create(
        model_path=str(SMPLX_MODELS),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    ).eval()


@torch.inference_mode()
def __joints_metres(model, name: str) -> np.ndarray:
    record = np.load(SOURCE / "smpl_on_ground" / f"{name}.npz")
    orient = np.asarray(record["global_orient"], dtype=np.float32)
    body = np.asarray(record["body_pose"], dtype=np.float32).reshape(-1, 23, 3)
    trans = np.asarray(record["transl"], dtype=np.float32)
    if not orient.shape[0] == body.shape[0] == trans.shape[0]:
        raise ValueError(f"{name}: frame counts differ {orient.shape} {body.shape} {trans.shape}")
    chunks = []
    for start in range(0, orient.shape[0], CHUNK_FRAMES):
        end = min(start + CHUNK_FRAMES, orient.shape[0])
        frames = end - start
        zero3 = torch.zeros(frames, 3)
        output = model(
            global_orient=torch.from_numpy(orient[start:end]),
            body_pose=torch.from_numpy(body[start:end, :21].reshape(frames, 63)),
            left_hand_pose=torch.zeros(frames, 45),
            right_hand_pose=torch.zeros(frames, 45),
            jaw_pose=zero3,
            leye_pose=zero3,
            reye_pose=zero3,
            expression=torch.zeros(frames, model.num_expression_coeffs),
            betas=torch.zeros(frames, 10),
            transl=torch.from_numpy(trans[start:end]),
        )
        chunks.append(output.joints[:, :BODY_JOINTS].numpy())
    joints = np.concatenate(chunks, axis=0)
    if not np.isfinite(joints).all():
        raise ValueError(f"{name}: non-finite joints")
    return joints.astype(np.float32)


def __y_up(joints_z_up: np.ndarray) -> np.ndarray:
    return np.stack((joints_z_up[..., 0], joints_z_up[..., 2], -joints_z_up[..., 1]), axis=-1)


def __foot_floor_p05_cm(joints_m: np.ndarray) -> float:
    foot_height = joints_m[:, [10, 11, 7, 8], 1].min(axis=1)
    return float(np.percentile(foot_height - foot_height.min(), 5) * CENTIMETRES_PER_METRE)


def __up_axis_ok(joints_m: np.ndarray) -> bool:
    head_minus_feet = joints_m[:, 15] - joints_m[:, [10, 11]].mean(axis=1)
    return bool(np.abs(head_minus_feet.mean(axis=0)).argmax() == 1)


def main() -> None:
    names = __selected_names()
    JOINTS_OUT.mkdir(parents=True, exist_ok=True)
    model = __create_model()
    rows = []
    for name in names:
        joints = __y_up(__joints_metres(model, name))
        np.save(JOINTS_OUT / f"{name}.npy", joints)
        rows.append(
            (
                name,
                joints.shape[0],
                round(joints.shape[0] / SOURCE_FPS, 2),
                __has_jump(name),
                round(__foot_floor_p05_cm(joints), 2),
                __up_axis_ok(joints),
            )
        )
    with open(INVENTORY, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(INVENTORY_COLUMNS)
        writer.writerows(rows)
    bad_axis = sum(1 for row in rows if not row[5])
    jumps = sum(1 for row in rows if row[3])
    print(f"clips {len(rows)}  with jump {jumps}  clips with wrong up axis {bad_axis}")
    sys.exit(1 if bad_axis else 0)


if __name__ == "__main__":
    main()
