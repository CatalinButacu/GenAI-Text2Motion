from __future__ import annotations

import numpy as np

from text2motion.motion.representation import FPS

FEET = [7, 8, 10, 11]
ACCELERATION_LIMIT_M_PER_FRAME2 = 0.08
TELEPORT_M_PER_FRAME = 0.25


def motion_statistics(joints: np.ndarray) -> dict[str, float]:
    velocity = np.diff(joints, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    root_step = np.linalg.norm(velocity[:, 0, [0, 2]], axis=-1)
    lowest = joints[:, FEET, 1].min(axis=-1)
    grounded = joints[:, FEET, 1] < np.percentile(lowest, 5) + 0.05
    foot_speed = np.linalg.norm(velocity[:, FEET][..., [0, 2]], axis=-1)
    skate = (foot_speed * grounded[1:]).sum() / max(1, grounded[1:].sum()) * FPS
    return {
        "mean_acceleration_cm_per_frame2": float(acceleration_norm.mean() * 100),
        "p99_acceleration_cm_per_frame2": float(np.percentile(acceleration_norm, 99) * 100),
        "mean_jerk_cm_per_frame3": float(np.linalg.norm(jerk, axis=-1).mean() * 100),
        "frames_over_acceleration_limit_pct": float(
            (acceleration_norm.max(axis=-1) > ACCELERATION_LIMIT_M_PER_FRAME2).mean() * 100
        ),
        "root_teleports": float((root_step > TELEPORT_M_PER_FRAME).sum()),
        "foot_skate_m_per_s": float(skate),
    }
