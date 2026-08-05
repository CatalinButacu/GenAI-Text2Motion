import numpy as np

from text2motion.data.hml3d.joints import kinematic_bones, recover_skeleton

__all__ = [
    "build_skeleton_seq",
    "kinematic_bones",
    "recover_skeleton",
    "render_skeleton_video",
    "view_skeleton",
]


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
