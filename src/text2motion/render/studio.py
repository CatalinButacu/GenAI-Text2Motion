import queue as queue_mod

import numpy as np
import torch

from text2motion.data.hml3d.feature import recover_from_ric
from text2motion.data.hml3d.param_util import joints_num, t2m_kinematic_chain
from text2motion.stream.decode import STREAM_END


def recover_skeleton(feat263: np.ndarray | torch.Tensor) -> np.ndarray:
    data = (
        feat263
        if isinstance(feat263, torch.Tensor)
        else torch.as_tensor(feat263, dtype=torch.float32)
    )
    return recover_from_ric(data.float(), joints_num).cpu().numpy()


def kinematic_bones() -> np.ndarray:
    bones = [
        [chain[i], chain[i + 1]] for chain in t2m_kinematic_chain for i in range(len(chain) - 1)
    ]
    return np.array(bones, dtype=np.int64)


def build_skeleton_seq(joints: np.ndarray):
    from aitviewer.renderables.skeletons import Skeletons  # lazy; raises if viewer extra absent

    return Skeletons(joint_positions=joints, joint_connections=kinematic_bones())


def view_skeleton(joints: np.ndarray) -> None:
    from aitviewer.viewer import Viewer  # lazy; raises if viewer extra absent

    viewer = Viewer()
    viewer.scene.add(build_skeleton_seq(joints))
    viewer.run()


def render_skeleton_video(joints: np.ndarray, out_path: str) -> str:
    from aitviewer.headless import HeadlessRenderer  # lazy; raises if viewer extra absent

    renderer = HeadlessRenderer()
    renderer.scene.add(build_skeleton_seq(joints))
    renderer.save_video(video_dir=out_path, output_fps=20)  # aitviewer>=1.14 API; 20fps = our rate
    return out_path


def collect_stream(out_queue: "queue_mod.Queue") -> np.ndarray:
    chunks: list[np.ndarray] = []

    while True:
        item = out_queue.get()

        if item is STREAM_END:
            break

        chunks.append(item.squeeze(0).cpu().numpy())  # (chunk_frames, 263), batch size 1

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 263), dtype=np.float32)
