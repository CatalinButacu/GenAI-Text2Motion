# .\.venv\Scripts\python.exe scripts/checks/body_model_error.py
import pathlib
import sys

import numpy as np
import torch

from text2motion.dataset.sources.humanml3d.build import joints_to_263
from text2motion.motion.representation import feature_params, target_offsets

REPO = pathlib.Path(__file__).resolve().parents[2]
CLIP = (
    REPO
    / "data"
    / "amass_work"
    / "KIT_raw"
    / "KIT"
    / "4"
    / "WalkInCounterClockwiseCircle02_stageii.npz"
)
MODELS = REPO / "data" / "smplx_models"
OFFICIAL = REPO / "data" / "HumanML3D_official" / "new_joint_vecs" / "000021.npy"
BODY_JOINTS = 22
TARGET_FPS = 20
SWAP_YZ = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def load_joints(
    gender: str, betas: np.ndarray, root: np.ndarray, body: np.ndarray, trans: np.ndarray
) -> np.ndarray:
    import smplx

    model = smplx.create(
        str(MODELS), model_type="smplx", gender=gender, use_pca=False, num_betas=10, ext="npz"
    )
    frames = body.shape[0]
    with torch.no_grad():
        out = model(
            betas=torch.tensor(betas[:10], dtype=torch.float32).expand(frames, -1),
            global_orient=torch.tensor(root, dtype=torch.float32),
            body_pose=torch.tensor(body, dtype=torch.float32),
            transl=torch.tensor(trans, dtype=torch.float32),
            left_hand_pose=torch.zeros(frames, 45),
            right_hand_pose=torch.zeros(frames, 45),
            jaw_pose=torch.zeros(frames, 3),
            leye_pose=torch.zeros(frames, 3),
            reye_pose=torch.zeros(frames, 3),
            expression=torch.zeros(frames, 10),
        )
    return out.joints[:, :BODY_JOINTS].numpy()


def stored_fps() -> float:
    import csv

    with open(REPO / "docs" / "notes" / "amass_clock_check.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source_path"].endswith("KIT/4/WalkInCounterClockwiseCircle02_poses.npy"):
                return float(row["stored_fps"])
    raise ValueError("recording not in clock table")


def main() -> None:
    clip = np.load(CLIP, allow_pickle=True)
    gender = str(clip["gender"])
    fps = stored_fps()
    step = round(fps / TARGET_FPS)
    root = clip["root_orient"][::step][:180]
    body = clip["pose_body"][::step][:180]
    trans = clip["trans"][::step][:180]
    betas = clip["betas"]
    joints = load_joints(gender, betas, root, body, trans)
    joints = joints @ SWAP_YZ.T
    joints[..., 0] *= -1
    params = feature_params(target_offsets(joints), TARGET_FPS)
    data, _ = joints_to_263(joints, params, BODY_JOINTS)
    official = np.load(OFFICIAL).astype(np.float32)
    frames = min(len(data), len(official))
    error = np.abs(data[:frames] - official[:frames])
    print(
        "gender",
        gender,
        "fps",
        fps,
        "step",
        step,
        "frames compared",
        frames,
        "official",
        len(official),
    )
    print("max abs error per feature block:")
    blocks = {
        "root": slice(0, 4),
        "positions": slice(4, 67),
        "rotations": slice(67, 193),
        "velocities": slice(193, 259),
        "contacts": slice(259, 263),
    }
    for name, sl in blocks.items():
        print(
            f"  {name:10s} max {float(error[:, sl].max()):.6f}  mean {float(error[:, sl].mean()):.6f}"
        )
    print("overall max", float(error.max()), "overall mean", float(error.mean()))
    sys.exit(0)


if __name__ == "__main__":
    main()
