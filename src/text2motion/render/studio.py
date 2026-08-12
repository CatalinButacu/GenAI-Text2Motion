import numpy as np

from text2motion.data.hml3d.joints import kinematic_bones, recover_skeleton

__all__ = [
    "build_skeleton_seq",
    "kinematic_bones",
    "recover_skeleton",
]


def build_skeleton_seq(joints: np.ndarray):
    from aitviewer.renderables.skeletons import Skeletons

    return Skeletons(joint_positions=joints, joint_connections=kinematic_bones())
