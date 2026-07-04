"""SMPL/T2M skeleton constants -- faithful port of the official HumanML3D code.

Constants copied EXACTLY (no edits to any value) from:
    https://github.com/EricGuo5513/HumanML3D/blob/main/paramUtil.py            (offsets, chains)
    https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb  (the
        ``__main__`` block: face_joint_indx, fid_l/fid_r, l_idx1/l_idx2, joints_num)
    https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb    (the
        ``swap_left_right`` mirror chains)

These define the 22-joint SMPL skeleton the standard-263 feature is built on; any change
breaks compatibility with the Guo evaluator and T2M-GPT's VQ-VAE. Do NOT touch the numbers.
"""

import numpy as np

# ---------------------------------------------------------------------------------------------
# 22-joint SMPL skeleton (the "t2m" skeleton used by HumanML3D-263)
# ---------------------------------------------------------------------------------------------

# Raw bone-direction offsets, one unit vector per joint (parent -> child direction).
t2m_raw_offsets = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ]
)

# Kinematic chains over the 22 body joints (root-first per chain).
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

# Hand chains (51-joint SMPL-H ordering) -- only used by the mirror swap when >24 joints present.
t2m_left_hand_chain = [
    [20, 22, 23, 24],
    [20, 34, 35, 36],
    [20, 25, 26, 27],
    [20, 31, 32, 33],
    [20, 28, 29, 30],
]
t2m_right_hand_chain = [
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 40, 41, 42],
    [21, 37, 38, 39],
    [21, 49, 50, 51],
]

# ---------------------------------------------------------------------------------------------
# process_file / regenerate constants (from motion_representation.ipynb __main__)
# ---------------------------------------------------------------------------------------------

# Lower-leg joints used to compute the uniform-skeleton scale ratio.
l_idx1, l_idx2 = 5, 8
# Right / left foot joint pairs for foot-contact detection.
fid_r, fid_l = [8, 11], [7, 10]
# Face direction reference joints: [r_hip, l_hip, sdr_r, sdr_l].
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1

joints_num = 22

# Foot-contact velocity threshold passed to process_file (squared-velocity cutoff).
feet_threshold = 0.002

# Reference skeleton clip id whose offsets define the uniform target skeleton.
t2m_tgt_skel_id = "000021"

# ---------------------------------------------------------------------------------------------
# Mirror augmentation chains (from raw_pose_processing.ipynb swap_left_right)
# ---------------------------------------------------------------------------------------------

mirror_right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
mirror_left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
mirror_left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
mirror_right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
