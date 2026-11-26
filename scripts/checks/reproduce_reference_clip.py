# .\.venv\Scripts\python.exe scripts/checks/reproduce_reference_clip.py
import pathlib
import sys

import numpy as np

from text2motion.dataset.sources.humanml3d.build import joints_to_263
from text2motion.motion.representation import feature_params, target_offsets

REPO = pathlib.Path(__file__).resolve().parents[2]
CLIP = (
    REPO
    / "data"
    / "amass_work"
    / "KIT_smplh"
    / "KIT"
    / "4"
    / "WalkInCounterClockwiseCircle02_poses.npz"
)
BODY = REPO / "data" / "body_models"
OFFICIAL = REPO / "data" / "HumanML3D_official" / "new_joint_vecs" / "000021.npy"
NUM_BETAS = 10
NUM_DMPLS = 8
TARGET_FPS = 20
START, END = 0, 180
JOINT_COUNT = 22
TRANS_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def load_model(gender: str) -> dict[str, np.ndarray]:
    data = np.load(BODY / "smplh" / gender / "model.npz", allow_pickle=True)
    parents = data["kintree_table"][0].astype(np.int64)
    parents[parents > 1000] = -1
    return {
        "v_template": data["v_template"],
        "shapedirs": data["shapedirs"][:, :, :NUM_BETAS],
        "J_regressor": data["J_regressor"],
        "parents": parents,
    }


def rest_joints(model: dict[str, np.ndarray], betas: np.ndarray) -> np.ndarray:
    v_shaped = model["v_template"] + model["shapedirs"] @ betas
    return model["J_regressor"] @ v_shaped


def pose_joints(clip: np.lib.npyio.NpzFile) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    gender = str(clip["gender"])
    fps = float(clip["mocap_framerate"])
    step = int(fps / TARGET_FPS)
    poses = clip["poses"][::step]
    trans = clip["trans"][::step]
    betas = clip["betas"][:NUM_BETAS]
    model = load_model(gender)
    rest = rest_joints(model, betas)
    parents = model["parents"]
    out = np.zeros((len(trans), JOINT_COUNT, 3))
    rotations = Rotation.from_rotvec(poses[:, : 3 * JOINT_COUNT].reshape(-1, 3)).as_matrix()
    rotations = rotations.reshape(len(trans), JOINT_COUNT, 3, 3)
    for frame in range(len(trans)):
        world_rot = [None] * JOINT_COUNT
        world_pos = [None] * JOINT_COUNT
        for j in range(JOINT_COUNT):
            local = rotations[frame, j]
            if parents[j] < 0:
                world_rot[j] = local
                world_pos[j] = rest[j]
            else:
                parent = parents[j]
                offset = rest[j] - rest[parent]
                world_rot[j] = world_rot[parent] @ local
                world_pos[j] = world_pos[parent] + world_rot[parent] @ offset
        out[frame] = np.stack(world_pos) + trans[frame]
    return np.dot(out, TRANS_MATRIX)


def main() -> None:
    clip = np.load(CLIP, allow_pickle=True)
    joints = pose_joints(clip)[START:END]
    joints[..., 0] *= -1
    reference = joints[:, :JOINT_COUNT]
    params = feature_params(target_offsets(reference), TARGET_FPS)
    data, _ = joints_to_263(reference, params, JOINT_COUNT)
    official = np.load(OFFICIAL).astype(np.float32)
    frames = min(len(data), len(official))
    error = np.abs(data[:frames] - official[:frames])
    print("joints", joints.shape, "features", data.shape, "official", official.shape)
    print("max abs error per block:")
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
    np.save(
        REPO / "data" / "amass_work" / "KIT_smplh" / "reproduced_000021_joints.npy",
        joints.astype(np.float32),
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
