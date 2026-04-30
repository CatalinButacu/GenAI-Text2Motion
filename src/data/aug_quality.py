from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.data.aug_slerp import ROTATION_SLICES


def geodesicJointVel(motion: np.ndarray, fps: float) -> float:
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
    maxVel = 0.0

    for sl in ROTATION_SLICES:
        aa = motion[:, sl]

        if aa.shape[1] != 3:
            continue
        rots = Rotation.from_rotvec(aa)
        rel = rots[:-1].inv() * rots[1:]
        # rel.magnitude() is in [0, pi]; division by dt gives rad/s
        v = float(rel.magnitude().max()) / dt

        if v > maxVel:
            maxVel = v

    return maxVel


def qualityFilter(
    motion: np.ndarray,
    fps: float = 30.0,
    *,
    maxAccel: float = 50.0,
    minFrames: int = 30,
    maxRootSpeed: float = 10.0,
    maxJointRotvel: float = 30.0,
    minVariance: float = 1e-4,
) -> bool:
    """Reject motion clips that are too short, corrupt, static, or kinematically implausible.

    Filters applied (in order, fail-fast):
      1. minFrames           -- shorter clips have no learning signal
      2. NaN/Inf reject      -- one bad sample wrecks normalisation stats
      3. fps sanity          -- impossible sources (<=1 Hz) cannot be resampled
      4. minVariance         -- whole-clip static (every channel near-constant) -> skip
      5. maxRootSpeed (m/s)  -- root translation teleport
      6. maxAccel (m/s^2)    -- root translation discontinuity
      7. maxJointRotvel rad/s -- per-joint geodesic angular velocity (mocap glitch).
                                 Computed on the rotation manifold so axis-angle
                                 wraparound is NOT counted as a snap.
    """
    T = motion.shape[0]

    if T < minFrames:
        return False

    if not np.isfinite(motion).all():
        return False

    if not np.isfinite(fps) or fps <= 1.0:
        return False

    if motion.var(axis=0).max() < minVariance:
        return False

    if motion.shape[1] >= 6:
        trans = motion[:, 3:6]
    else:
        return True

    dt = 1.0 / fps
    velocity = np.diff(trans, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=1)

    if speed.max() > maxRootSpeed:
        return False

    if T >= 3:
        accel = np.diff(velocity, axis=0) / dt

        if np.linalg.norm(accel, axis=1).max() > maxAccel:
            return False

    if motion.shape[1] >= 9:
        if geodesicJointVel(motion, fps) > maxJointRotvel:
            return False

    return True


def detectTpose(
    motion: np.ndarray,
    *,
    varianceThreshold: float = 0.001,
    nCheckFrames: int = 10,
) -> tuple[int, int]:
    T = motion.shape[0]

    if T <= nCheckFrames * 2:
        return 0, 0

    trimStart = 0

    for i in range(min(nCheckFrames, T)):
        if motion[i].var() < varianceThreshold:
            trimStart = i + 1
        else:
            break

    trimEnd = 0

    for i in range(T - 1, max(T - nCheckFrames - 1, trimStart), -1):
        if motion[i].var() < varianceThreshold:
            trimEnd = T - i
        else:
            break

    return trimStart, trimEnd
