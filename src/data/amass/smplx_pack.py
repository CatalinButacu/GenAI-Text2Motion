from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SMPLXSample:
    sample_id: str
    motion: np.ndarray  # (T, 168)
    betas: np.ndarray  # (16,)
    fps: float
    duration: float
    gender: str = "neutral"
    text: str = ""
    source: str = "amass"
    object_motion: np.ndarray | None = None  # (T, 7) for ARCTIC


def pack_smplx_pose(
    root_orient: np.ndarray,
    trans: np.ndarray,
    pose_body: np.ndarray,
    pose_hand: np.ndarray,
    pose_jaw: np.ndarray,
    pose_eye: np.ndarray,
) -> np.ndarray:
    T = root_orient.shape[0]
    assert pose_hand.shape == (T, 90), f"pose_hand: expected ({T},90), got {pose_hand.shape}"
    result = np.concatenate(
        [root_orient, trans, pose_body, pose_hand, pose_jaw, pose_eye], axis=1,
    ).astype(np.float32)
    assert result.shape[1] == 168, f"packed pose: expected 168 dims, got {result.shape[1]}"

    return result
