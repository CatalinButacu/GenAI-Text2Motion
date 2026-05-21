from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.data.aug_slerp import ROTATION_SLICES


def geodesic_joint_vel(motion: np.ndarray, fps: float) -> float:
    """Maximum per-joint geodesic angular velocity (rad/s) across the clip.

    Uses the rotation manifold rather than per-channel differences, so it does
    NOT trigger on the +pi <-> -pi wraparound that plain np.diff(axis-angle)
    sees as a ~2pi/dt "snap" (~188 rad/s @ 30 fps -- a representation
    artefact, not real motion).

    For each rotation joint slice (root_orient + 54 body/hand/face joints)
    we compute the relative rotation R_t.inv() * R_{t+1} and take its
    magnitude (geodesic distance in [0, pi]). Velocity = magnitude / dt.

    Returns 0.0 when the clip has fewer than 2 frames or 9 channels.
    """
    if motion.shape[0] < 2 or motion.shape[1] < 9:
        return 0.0

    dt = 1.0 / fps
    max_vel = 0.0

    for sl in ROTATION_SLICES:
        aa = motion[:, sl]

        if aa.shape[1] != 3:
            continue
        rots = Rotation.from_rotvec(aa)
        rel = rots[:-1].inv() * rots[1:]
        # rel.magnitude() is in [0, pi]; division by dt gives rad/s
        v = float(rel.magnitude().max()) / dt

        if v > max_vel:
            max_vel = v

    return max_vel


def quality_filter(
    motion: np.ndarray,
    fps: float = 30.0,
    *,
    max_accel: float = 50.0,
    min_frames: int = 30,
    max_root_speed: float = 10.0,
    max_joint_rotvel: float = 30.0,
    min_variance: float = 1e-4,
) -> bool:
    """Reject motion clips that are too short, corrupt, static, or kinematically implausible.

    Filters applied (in order, fail-fast):
      1. min_frames           -- shorter clips have no learning signal
      2. NaN/Inf reject      -- one bad sample wrecks normalisation stats
      3. fps sanity          -- impossible sources (<=1 Hz) cannot be resampled
      4. min_variance         -- whole-clip static (every channel near-constant) -> skip
      5. max_root_speed (m/s)  -- root translation teleport
      6. max_accel (m/s^2)    -- root translation discontinuity
      7. max_joint_rotvel rad/s -- per-joint geodesic angular velocity (mocap glitch).
                                 Computed on the rotation manifold so axis-angle
                                 wraparound is NOT counted as a snap.
    """
    T = motion.shape[0]

    if T < min_frames:
        return False

    if not np.isfinite(motion).all():
        return False

    if not np.isfinite(fps) or fps <= 1.0:
        return False

    if motion.var(axis=0).max() < min_variance:
        return False

    if motion.shape[1] >= 6:
        trans = motion[:, 3:6]
    else:
        return True

    dt = 1.0 / fps
    velocity = np.diff(trans, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=1)

    if speed.max() > max_root_speed:
        return False

    if T >= 3:
        accel = np.diff(velocity, axis=0) / dt

        if np.linalg.norm(accel, axis=1).max() > max_accel:
            return False

    if motion.shape[1] >= 9:
        if geodesic_joint_vel(motion, fps) > max_joint_rotvel:
            return False

    return True


def detect_tpose(
    motion: np.ndarray,
    *,
    variance_threshold: float = 0.001,
    n_check_frames: int = 10,
) -> tuple[int, int]:
    T = motion.shape[0]

    if T <= n_check_frames * 2:
        return 0, 0

    trim_start = 0

    for i in range(min(n_check_frames, T)):
        if motion[i].var() < variance_threshold:
            trim_start = i + 1
        else:
            break

    trim_end = 0

    for i in range(T - 1, max(T - n_check_frames - 1, trim_start), -1):
        if motion[i].var() < variance_threshold:
            trim_end = T - i
        else:
            break

    return trim_start, trim_end
