"""aitviewer studio for the 263 (body-only) track: recover the 22-joint skeleton from the feature
and view it interactively, render it headless to MP4, or play a live stream of frame chunks.

The 263 feature is recovered to joint POSITIONS (`recover_from_ric`) and shown as an aitviewer
`Skeletons` renderable with the t2m kinematic chain. aitviewer is the optional `[viewer]` extra and
is imported lazily, so this module imports without it. See `.claude/skills/aitviewer-studio`.
(The 168 SMPL-X track would instead use aitviewer's SMPLSequence -- deferred.)
"""

import queue as queue_mod

import numpy as np
import torch

from text2motion.data.hml3d.feature import recover_from_ric
from text2motion.data.hml3d.param_util import joints_num, t2m_kinematic_chain
from text2motion.stream.decode import STREAM_END


def recover_skeleton(feat263: np.ndarray | torch.Tensor) -> np.ndarray:
    """(T, 263) or (B, T, 263) -> joint positions (T, 22, 3) / (B, T, 22, 3)."""
    data = (
        feat263
        if isinstance(feat263, torch.Tensor)
        else torch.as_tensor(feat263, dtype=torch.float32)
    )
    return recover_from_ric(data.float(), joints_num).cpu().numpy()


def kinematic_bones() -> np.ndarray:
    """Bone index pairs (E, 2) from the t2m kinematic chains (for the Skeletons renderable)."""
    bones = [
        [chain[i], chain[i + 1]] for chain in t2m_kinematic_chain for i in range(len(chain) - 1)
    ]
    return np.array(bones, dtype=np.int64)


def build_skeleton_seq(joints: np.ndarray):
    """Build an aitviewer Skeletons renderable from (T, 22, 3) joint positions."""
    from aitviewer.renderables.skeletons import Skeletons  # lazy; raises if viewer extra absent

    return Skeletons(joint_positions=joints, joint_connections=kinematic_bones())


def view_skeleton(joints: np.ndarray) -> None:
    """Open the interactive studio on a recovered (T, 22, 3) clip."""
    from aitviewer.viewer import Viewer  # lazy; raises if viewer extra absent

    viewer = Viewer()
    viewer.scene.add(build_skeleton_seq(joints))
    viewer.run()


def render_skeleton_video(joints: np.ndarray, out_path: str) -> str:
    """Headless-render a (T, 22, 3) clip to MP4. Needs a working GL/EGL context."""
    from aitviewer.headless import HeadlessRenderer  # lazy; raises if viewer extra absent

    renderer = HeadlessRenderer()
    renderer.scene.add(build_skeleton_seq(joints))
    renderer.save_video(video_dir=out_path, output_fps=20)  # aitviewer>=1.14 API; 20fps = our rate
    return out_path


def collect_stream(out_queue: "queue_mod.Queue") -> np.ndarray:
    """Drain a producer queue (from stream.decode.run_producer) until STREAM_END into one
    (T, 263) array. Use for headless render of a streamed clip; for true live playback hook the
    chunks into the Viewer's per-frame update callback instead."""
    chunks: list[np.ndarray] = []

    while True:
        item = out_queue.get()

        if item is STREAM_END:
            break

        chunks.append(item.squeeze(0).cpu().numpy())  # (chunk_frames, 263), batch size 1

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 263), dtype=np.float32)
