from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from text2motion.motion.recovery import (
    recover_from_ric,
)
from text2motion.motion.representation import (
    JOINTS,
    FeatureParams,
    encode_joints,
    feature_params,
    target_offsets,
)

LEFT_ANKLE, LEFT_FOOT = 7, 10


def encoder_params(reference_joints: Path, fps: int) -> FeatureParams:
    reference = np.load(reference_joints)[:, :JOINTS].reshape(-1, JOINTS, 3)
    return feature_params(target_offsets(reference), fps)


def clip_features(joints: np.ndarray, params: FeatureParams) -> np.ndarray | None:
    features = encode_joints(joints, params)[0].astype(np.float32)
    return features if np.isfinite(features).all() else None


def feature_joints(features: np.ndarray) -> np.ndarray:
    if not np.isfinite(features).all():
        raise ValueError("features hold NaN or Inf")
    return recover_from_ric(torch.from_numpy(features).float(), JOINTS).numpy()


def feet_point_forward(features: np.ndarray) -> bool:
    joints = feature_joints(features[:1])[0]
    return bool(joints[LEFT_FOOT, 2] - joints[LEFT_ANKLE, 2] > 0)
