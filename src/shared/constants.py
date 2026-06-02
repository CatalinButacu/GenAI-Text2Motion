import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    motion_fps: int
    default_clip_duration_s: float


@dataclass(frozen=True, slots=True)
class TokenSpec:
    pad_token_id: int
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    special_tokens: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class SceneSpec:
    colors: Mapping[str, tuple[float, float, float, float]]


@dataclass(frozen=True, slots=True)
class SmplxSpec:
    n_joints: int
    n_verts: int
    n_body_joints: int
    n_hand_joints: int
    pose_dim: int
    root_orient_slice: slice
    transl_slice: slice
    body_pose_slice: slice
    lhand_pose_slice: slice
    rhand_pose_slice: slice
    transl_y_idx: int
    transl_z_idx: int
    skeleton: dict[int, tuple[str, int]]
    joint_names: list[str]
    parents: list[int]
    body_bones: list[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class SharedConstants:
    runtime: RuntimeSpec
    tokens: TokenSpec
    scene: SceneSpec
    smplx: SmplxSpec


def build_runtime_spec() -> RuntimeSpec:
    return RuntimeSpec(motion_fps=30, default_clip_duration_s=5.0)


def build_token_spec() -> TokenSpec:
    special_tokens = MappingProxyType({"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3})

    return TokenSpec(
        pad_token_id=special_tokens["<PAD>"],
        unk_token_id=special_tokens["<UNK>"],
        bos_token_id=special_tokens["<BOS>"],
        eos_token_id=special_tokens["<EOS>"],
        special_tokens=special_tokens,
    )


def build_scene_spec() -> SceneSpec:
    return SceneSpec(
        colors=MappingProxyType(
            {
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
        )
    )


def build_smplx_spec() -> SmplxSpec:
    skeleton: dict[int, tuple[str, int]] = {
        0: ("pelvis", -1),
        1: ("left_hip", 0),
        2: ("right_hip", 0),
        3: ("spine1", 0),
        4: ("left_knee", 1),
        5: ("right_knee", 2),
        6: ("spine2", 3),
        7: ("left_ankle", 4),
        8: ("right_ankle", 5),
        9: ("spine3", 6),
        10: ("left_foot", 7),
        11: ("right_foot", 8),
        12: ("neck", 9),
        13: ("left_collar", 9),
        14: ("right_collar", 9),
        15: ("head", 12),
        16: ("left_shoulder", 13),
        17: ("right_shoulder", 14),
        18: ("left_elbow", 16),
        19: ("right_elbow", 17),
        20: ("left_wrist", 18),
        21: ("right_wrist", 19),
        22: ("L_index1", 20),
        23: ("L_index2", 22),
        24: ("L_index3", 23),
        25: ("L_middle1", 20),
        26: ("L_middle2", 25),
        27: ("L_middle3", 26),
        28: ("L_pinky1", 20),
        29: ("L_pinky2", 28),
        30: ("L_pinky3", 29),
        31: ("L_ring1", 20),
        32: ("L_ring2", 31),
        33: ("L_ring3", 32),
        34: ("L_thumb1", 20),
        35: ("L_thumb2", 34),
        36: ("L_thumb3", 35),
        37: ("R_index1", 21),
        38: ("R_index2", 37),
        39: ("R_index3", 38),
        40: ("R_middle1", 21),
        41: ("R_middle2", 40),
        42: ("R_middle3", 41),
        43: ("R_pinky1", 21),
        44: ("R_pinky2", 43),
        45: ("R_pinky3", 44),
        46: ("R_ring1", 21),
        47: ("R_ring2", 46),
        48: ("R_ring3", 47),
        49: ("R_thumb1", 21),
        50: ("R_thumb2", 49),
        51: ("R_thumb3", 50),
        52: ("jaw", 15),
        53: ("left_eye", 15),
        54: ("right_eye", 15),
    }
    n_joints = 55
    n_body_joints = 22

    return SmplxSpec(
        n_joints=n_joints,
        n_verts=10475,
        n_body_joints=n_body_joints,
        n_hand_joints=15,
        pose_dim=168,
        root_orient_slice=slice(0, 3),
        transl_slice=slice(3, 6),
        body_pose_slice=slice(6, 69),
        lhand_pose_slice=slice(69, 114),
        rhand_pose_slice=slice(114, 159),
        transl_y_idx=4,
        transl_z_idx=5,
        skeleton=skeleton,
        joint_names=[skeleton[i][0] for i in range(n_joints)],
        parents=[skeleton[i][1] for i in range(n_joints)],
        body_bones=[(skeleton[c][1], c) for c in range(n_body_joints) if skeleton[c][1] != -1],
    )


SMPLX = build_smplx_spec()


CONSTS = SharedConstants(
    runtime=build_runtime_spec(),
    tokens=build_token_spec(),
    scene=build_scene_spec(),
    smplx=SMPLX,
)


# ── SSM architecture ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SsmSpec:
    d_model: int
    d_state: int
    n_layers: int


SSM = SsmSpec(d_model=384, d_state=64, n_layers=6)


# ── Mamba SSM cell hyperparams ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MambaSpec:
    d_conv: int
    expand: int
    dt_min: float
    dt_max: float
    dt_rank_divisor: int
    dt_clamp_min: float


MAMBA = MambaSpec(
    d_conv=4,
    expand=2,
    dt_min=0.001,
    dt_max=0.1,
    dt_rank_divisor=16,
    dt_clamp_min=1e-4,
)


# ── RVQ tokenizer hyperparams ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RvqSpec:
    codebook_init_scale: float
    ema_decay: float
    ema_eps: float
    dead_code_reset_threshold: float
    dead_code_noise_std: float
    active_code_threshold: float
    valid_down_t: tuple[int, ...]


RVQ = RvqSpec(
    codebook_init_scale=0.01,
    ema_decay=0.99,
    ema_eps=1e-5,
    dead_code_reset_threshold=1.0,
    dead_code_noise_std=0.01,
    active_code_threshold=0.5,
    valid_down_t=(1, 2, 4, 8),
)


# ── Training defaults ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    min_quality_frames: int
    cache_hash_len: int
    joblib_compress_level: int
    default_unified_sources: tuple[str, ...]
    default_amass_dir: str
    default_arctic_dir: str
    default_humanml3d_dir: str
    default_interx_dir: str


TRAINING = TrainingSpec(
    min_quality_frames=30,
    cache_hash_len=12,
    joblib_compress_level=3,
    default_unified_sources=("amass", "arctic"),
    default_amass_dir="data/AMASS",
    default_arctic_dir="data/arctic/unpack",
    default_humanml3d_dir="data/humanml3d",
    default_interx_dir="data/inter-x",
)


# ── Evaluation thresholds ─────────────────────────────────────────────────────

Z_95 = 1.96

T_CRITICAL: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

OUTLIER_SIGMA = 3.0
RNG_SEED = 42
SEMANTIC_KEYWORDS_LIMIT = 6
SEMANTIC_MATCH_THRESHOLD = 0.45
BEST_CHECKPOINT_MARKER = "best_model.pt"
LAST_CHECKPOINT_MARKER = "last_model.pt"


# ── Understanding constants ───────────────────────────────────────────────────

SEQUENCE_MARKERS: list[tuple[str, bool]] = [
    ("and then", False),
    ("then", False),
    ("subsequently", False),
    ("after that", False),
    ("followed by", False),
    ("next", False),
    ("before", False),
    ("first", False),
    ("simultaneously", True),
    ("while", True),
    ("at the same time", True),
    ("in parallel", True),
    ("concurrently", True),
]

DURATION_RE = re.compile(
    r"(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|second|min|minute|hour|h)?",
    re.IGNORECASE,
)

MODIFIER_WORDS = frozenset(
    {
        "quickly",
        "slowly",
        "fast",
        "aggressively",
        "nervously",
        "happily",
        "tiredly",
        "carefully",
        "gently",
        "violently",
        "angrily",
        "stressed",
        "anxiously",
        "calmly",
        "frantically",
        "lazily",
        "gracefully",
        "awkwardly",
        "smoothly",
        "hesitantly",
        "confidently",
        "fearfully",
    }
)

SPATIAL_RELATIONS: Mapping[str, str] = MappingProxyType(
    {
        "on": "ON",
        "on top of": "ON",
        "above": "ABOVE",
        "over": "ABOVE",
        "below": "BELOW",
        "under": "BELOW",
        "beneath": "BELOW",
        "in front of": "IN_FRONT_OF",
        "in front": "IN_FRONT_OF",
        "before": "IN_FRONT_OF",
        "behind": "BEHIND",
        "back of": "BEHIND",
        "to the left": "TO_LEFT",
        "left of": "TO_LEFT",
        "to the right": "TO_RIGHT",
        "right of": "TO_RIGHT",
        "near": "NEAR",
        "next to": "NEAR",
        "beside": "NEAR",
        "close to": "NEAR",
        "far from": "FAR",
        "inside": "IN",
        "in": "IN",
        "within": "IN",
        "outside": "OUT",
        "between": "BETWEEN",
        "among": "AMONG",
    }
)

WORD_TO_COUNT: Mapping[str, int] = MappingProxyType(
    {
        "zero": 0,
        "one": 1,
        "two": 2,
        "a couple": 2,
        "couple": 2,
        "three": 3,
        "a few": 3,
        "few": 3,
        "four": 4,
        "several": 4,
        "five": 5,
        "many": 6,
        "multiple": 3,
    }
)

ANAPHORIC_HUMANOID = frozenset(
    {
        "another",
        "other",
        "others",
        "someone",
        "they",
        "them",
        "their",
        "he",
        "she",
        "it",
        "one",
        "the person",
        "the actor",
        "the human",
    }
)

RULER_NAME = "vocab_entity_ruler"


# ── Data constants ────────────────────────────────────────────────────────────

CACHE_SCHEMA = 4
INGEST_MAX_LENGTH = 1000

DEFAULT_FILTER_KWARGS: dict[str, int | float] = {
    "min_frames": 30,
    "max_root_speed": 10.0,
    "max_accel": 50.0,
    "max_joint_rotvel": 30.0,
    "min_variance": 1e-4,
}

SKIP_ACTIONS = frozenset(
    {
        "male",
        "female",
        "female1",
        "male1",
        "male2",
        "female2",
        "subj calibration",
    }
)

AMASS_FPS = 30
INTERX_FPS = 30.0
ACTION_CODE_RE = re.compile(r"A(\d{3})")

LR_WORD_RE = re.compile(r"\b(left|right|left-hand|right-hand|l_|r_)\b", re.IGNORECASE)

LR_SWAP: dict[str, str] = {
    "left": "right",
    "right": "left",
    "left-hand": "right-hand",
    "right-hand": "left-hand",
    "l_": "r_",
    "r_": "l_",
}

HUMANML3D_DEFAULT_DIR = "data/humanml3d"
HUMANML3D_METADATA_RE = re.compile(r"#.*$")
UNIFIED_SOURCE_NAMES = ("amass", "arctic", "humanml3d", "inter-x")


# ── Render constants ──────────────────────────────────────────────────────────

R_ZUP_TO_YUP = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

R_IDENTITY = np.eye(3, dtype=np.float32)
CAMERA_POSITION = np.array([0.0, 1.5, 4.5], dtype=np.float32)
CAMERA_TARGET = np.array([0.0, 1.0, 0.0], dtype=np.float32)
BACKGROUND_COLOR = np.array([0.85, 0.87, 0.90, 1.0], dtype=np.float32)
FLOOR_SIZE = 100.0
FLOOR_DIVISIONS = 200
FLOOR_PRIMARY = (0.82, 0.83, 0.84, 1.0)
FLOOR_SECONDARY = (0.80, 0.81, 0.82, 1.0)
FLOOR_PLANE = "xz"
SKIN_COLOR = (0.72, 0.60, 0.52, 1.0)
SMPLX_DIR = "data/arctic/unpack/models"
