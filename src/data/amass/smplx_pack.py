from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SMPLXSample:
    sampleId: str
    motion: np.ndarray  # (T, 168)
    betas: np.ndarray  # (16,)
    fps: float
    duration: float
    gender: str = "neutral"
    text: str = ""
    source: str = "amass"
    objectMotion: np.ndarray | None = None  # (T, 7) for ARCTIC


def packSmplxPose(
    rootOrient: np.ndarray,
    trans: np.ndarray,
    poseBody: np.ndarray,
    poseHand: np.ndarray,
    poseJaw: np.ndarray,
    poseEye: np.ndarray,
) -> np.ndarray:
    T = rootOrient.shape[0]
    assert poseHand.shape == (T, 90), f"poseHand: expected ({T},90), got {poseHand.shape}"
    result = np.concatenate(
        [rootOrient, trans, poseBody, poseHand, poseJaw, poseEye], axis=1,
    ).astype(np.float32)
    assert result.shape[1] == 168, f"packed pose: expected 168 dims, got {result.shape[1]}"

    return result
