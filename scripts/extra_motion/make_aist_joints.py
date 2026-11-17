# .\.venv\Scripts\python.exe scripts/extra_motion/make_aist_joints.py
import csv
import pathlib
import pickle
import sys
import zipfile

import numpy as np
import smplx
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE_ZIP = REPO / "data" / "extra_motion" / "aist" / "motions.zip"
UNZIPPED = REPO / "data" / "extra_motion" / "aist" / "unzipped"
JOINTS_OUT = REPO / "data" / "extra_motion" / "joints60" / "aist"
INVENTORY = REPO / "data" / "extra_motion" / "inventory_aist.csv"
SMPLX_MODELS = REPO / "data" / "smplx_models"
SOURCE_FPS = 60
BODY_JOINTS = 22
CHUNK_FRAMES = 256
CENTIMETRES_PER_METRE = 100.0
ALLOWED_GLOBALS = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }
)
INVENTORY_COLUMNS = ("name", "frames", "seconds", "scaling", "foot_floor_p05_cm", "up_axis_ok")


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def __unzip() -> list[pathlib.Path]:
    if not UNZIPPED.exists():
        with zipfile.ZipFile(SOURCE_ZIP) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith(".pkl") and "__MACOSX" not in name
            ]
            archive.extractall(UNZIPPED, members)
    files = sorted(UNZIPPED.rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no pkl files under {UNZIPPED}")
    return files


def __load_record(path: pathlib.Path) -> dict:
    with open(path, "rb") as handle:
        record = _SafeUnpickler(handle).load()
    missing = {"smpl_poses", "smpl_trans", "smpl_scaling"} - set(record)
    if missing:
        raise KeyError(f"{path.name}: missing keys {sorted(missing)}")
    return record


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
def __joints_metres(model, record: dict) -> tuple[np.ndarray, float]:
    poses = np.asarray(record["smpl_poses"], dtype=np.float32).reshape(-1, 24, 3)
    trans = np.asarray(record["smpl_trans"], dtype=np.float32)
    scaling = float(np.asarray(record["smpl_scaling"]).ravel()[0])
    if poses.shape[0] != trans.shape[0] or scaling <= 0.0:
        raise ValueError(f"bad shapes {poses.shape} {trans.shape} or scaling {scaling}")
    chunks = []
    for start in range(0, poses.shape[0], CHUNK_FRAMES):
        end = min(start + CHUNK_FRAMES, poses.shape[0])
        frames = end - start
        pose = torch.from_numpy(poses[start:end])
        zero3 = torch.zeros(frames, 3)
        output = model(
            global_orient=pose[:, 0],
            body_pose=pose[:, 1:22].reshape(frames, 63),
            left_hand_pose=torch.zeros(frames, 45),
            right_hand_pose=torch.zeros(frames, 45),
            jaw_pose=zero3,
            leye_pose=zero3,
            reye_pose=zero3,
            expression=torch.zeros(frames, model.num_expression_coeffs),
            betas=torch.zeros(frames, 10),
            transl=torch.from_numpy(trans[start:end] / scaling),
        )
        chunks.append(output.joints[:, :BODY_JOINTS].numpy())
    joints = np.concatenate(chunks, axis=0) * scaling / CENTIMETRES_PER_METRE
    if not np.isfinite(joints).all():
        raise ValueError("non-finite joints")
    return joints.astype(np.float32), scaling


def __foot_floor_p05_cm(joints_m: np.ndarray) -> float:
    foot_height = joints_m[:, [10, 11, 7, 8], 1].min(axis=1)
    return float(np.percentile(foot_height - foot_height.min(), 5) * CENTIMETRES_PER_METRE)


def __up_axis_ok(joints_m: np.ndarray) -> bool:
    head_minus_feet = joints_m[:, 15] - joints_m[:, [10, 11]].mean(axis=1)
    return bool(np.abs(head_minus_feet.mean(axis=0)).argmax() == 1)


def main() -> None:
    files = __unzip()
    JOINTS_OUT.mkdir(parents=True, exist_ok=True)
    model = __create_model()
    rows = []
    for path in files:
        joints, scaling = __joints_metres(model, __load_record(path))
        np.save(JOINTS_OUT / f"{path.stem}.npy", joints)
        rows.append(
            (
                path.stem,
                joints.shape[0],
                round(joints.shape[0] / SOURCE_FPS, 2),
                round(scaling, 3),
                round(__foot_floor_p05_cm(joints), 2),
                __up_axis_ok(joints),
            )
        )
    with open(INVENTORY, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(INVENTORY_COLUMNS)
        writer.writerows(rows)
    hours = sum(row[1] for row in rows) / SOURCE_FPS / 3600
    bad_axis = sum(1 for row in rows if not row[5])
    print(f"clips {len(rows)}  hours {hours:.2f}  clips with wrong up axis {bad_axis}")
    sys.exit(1 if bad_axis else 0)


if __name__ == "__main__":
    main()
