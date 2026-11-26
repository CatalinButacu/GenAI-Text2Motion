import numpy as np

from text2motion.evaluation.metrics import FEET, jerk_profile
from text2motion.motion.representation import FPS, kinematic_bones


def require_transition_window(joints: np.ndarray, boundary: int, half_window: int) -> None:
    if half_window < 2 or boundary < half_window or boundary + half_window > len(joints):
        raise ValueError("transition window does not fit the motion")


def transition_bone_deviation(data: np.ndarray, boundary: int, half_window: int) -> np.ndarray:
    bones = kinematic_bones()
    lengths = np.linalg.norm(data[:, bones[:, 1]] - data[:, bones[:, 0]], axis=-1)
    outside = np.concatenate((lengths[: boundary - half_window], lengths[boundary + half_window :]))
    if not len(outside):
        raise ValueError("bone reference needs frames outside the transition window")
    reference = np.median(outside, axis=0)
    if (reference <= 1e-8).any():
        raise ValueError("transition has a degenerate reference bone")
    return np.abs(lengths[boundary - half_window : boundary + half_window] / reference - 1)


def transition_statistics(
    joints: np.ndarray, boundary: int, half_window: int = 30, floor_height: float = 0.0
) -> dict[str, float | None]:
    jerk_profile(joints)
    require_transition_window(joints, boundary, half_window)
    if not np.isfinite(floor_height):
        raise ValueError("the common floor height must be finite")
    data = np.asarray(joints, dtype=np.float64)
    window = data[boundary - half_window : boundary + half_window]
    feet = window[:, FEET]
    contact = feet[1:, :, 1] <= floor_height + 0.05
    speed = np.linalg.norm(np.diff(feet[..., [0, 2]], axis=0), axis=-1) * FPS
    relative = transition_bone_deviation(data, boundary, half_window)
    return {
        "foot_skate_m_per_s": float(speed[contact].mean()) if contact.any() else None,
        "foot_contact_fraction": float(contact.mean()),
        "feet_below_floor_2cm_fraction": float((feet[..., 1] < floor_height - 0.02).mean()),
        "all_feet_above_floor_10cm_fraction": float(
            (feet[..., 1].min(axis=-1) > floor_height + 0.1).mean()
        ),
        "bone_relative_deviation": float(relative.mean()),
        "bone_relative_deviation_max": float(relative.max()),
    }
