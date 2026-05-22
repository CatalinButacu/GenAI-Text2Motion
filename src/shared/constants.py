SMPLX_N_JOINTS: int = 55
N_BODY_JOINTS: int = 22
N_HAND_JOINTS: int = 15
SMPLX_N_VERTS: int = 10475

MOTION_DIM: int = 168
MOTION_FPS: int = 30

SMPLX_ROOT_ORIENT_SLICE = slice(0, 3)
SMPLX_TRANSL_SLICE = slice(3, 6)
SMPLX_BODY_POSE_SLICE = slice(6, 69)
SMPLX_TRANSL_Y_IDX: int = 4  # horizontal Y component of translation (front-back in Z-up AMASS)
SMPLX_TRANSL_Z_IDX: int = 5  # vertical Z component of translation (up-axis in Z-up AMASS data)

PAD_TOKEN_ID: int = 0
UNK_TOKEN_ID: int = 1
BOS_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 3
SPECIAL_TOKENS: dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

# Defaults match configs/motion_ssm.yaml (the cloud headline run).
# tests/test_config_drift.py asserts these stay in lockstep with the YAML.
SSM_D_MODEL: int = 384
SSM_D_STATE: int = 64
SSM_N_LAYERS: int = 6

# --- Rendering / scene colours ---
SCENE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (1.0, 0.1, 0.1, 1.0),
    "green": (0.1, 0.8, 0.1, 1.0),
    "blue": (0.1, 0.1, 1.0, 1.0),
    "yellow": (1.0, 0.9, 0.0, 1.0),
    "orange": (1.0, 0.5, 0.0, 1.0),
    "purple": (0.5, 0.0, 0.8, 1.0),
    "white": (0.95, 0.95, 0.95, 1.0),
    "black": (0.1, 0.1, 0.1, 1.0),
    "gray": (0.5, 0.5, 0.5, 1.0),
    "grey": (0.5, 0.5, 0.5, 1.0),
    "brown": (0.5, 0.3, 0.1, 1.0),
    "pink": (1.0, 0.5, 0.7, 1.0),
    "cyan": (0.0, 0.8, 0.8, 1.0),
}

# --- Skeleton topology ---
# Single source of truth: joint_index -> (name, parent_index).  Parent -1 = root.
SMPLX_SKELETON: dict[int, tuple[str, int]] = {
     0: ("pelvis",          -1),
     1: ("left_hip",         0),
     2: ("right_hip",        0),
     3: ("spine1",           0),
     4: ("left_knee",        1),
     5: ("right_knee",       2),
     6: ("spine2",           3),
     7: ("left_ankle",       4),
     8: ("right_ankle",      5),
     9: ("spine3",           6),
    10: ("left_foot",        7),
    11: ("right_foot",       8),
    12: ("neck",             9),
    13: ("left_collar",      9),
    14: ("right_collar",     9),
    15: ("head",            12),
    16: ("left_shoulder",   13),
    17: ("right_shoulder",  14),
    18: ("left_elbow",      16),
    19: ("right_elbow",     17),
    20: ("left_wrist",      18),
    21: ("right_wrist",     19),
    22: ("L_index1",        20),
    23: ("L_index2",        22),
    24: ("L_index3",        23),
    25: ("L_middle1",       20),
    26: ("L_middle2",       25),
    27: ("L_middle3",       26),
    28: ("L_pinky1",        20),
    29: ("L_pinky2",        28),
    30: ("L_pinky3",        29),
    31: ("L_ring1",         20),
    32: ("L_ring2",         31),
    33: ("L_ring3",         32),
    34: ("L_thumb1",        20),
    35: ("L_thumb2",        34),
    36: ("L_thumb3",        35),
    37: ("R_index1",        21),
    38: ("R_index2",        37),
    39: ("R_index3",        38),
    40: ("R_middle1",       21),
    41: ("R_middle2",       40),
    42: ("R_middle3",       41),
    43: ("R_pinky1",        21),
    44: ("R_pinky2",        43),
    45: ("R_pinky3",        44),
    46: ("R_ring1",         21),
    47: ("R_ring2",         46),
    48: ("R_ring3",         47),
    49: ("R_thumb1",        21),
    50: ("R_thumb2",        49),
    51: ("R_thumb3",        50),
    52: ("jaw",             15),
    53: ("left_eye",        15),
    54: ("right_eye",       15),
}

# Derived flat views — computed once from SMPLX_SKELETON.
SMPLX_JOINT_NAMES: list[str]         = [SMPLX_SKELETON[i][0] for i in range(SMPLX_N_JOINTS)]
SMPLX_PARENTS: list[int]             = [SMPLX_SKELETON[i][1] for i in range(SMPLX_N_JOINTS)]
SMPLX_BODY_BONES: list[tuple[int, int]] = [
    (SMPLX_SKELETON[c][1], c)
    for c in range(N_BODY_JOINTS)
    if SMPLX_SKELETON[c][1] != -1
]
