"""Convert our SMPL-X 168-dim motions to Inter-X canonical (T, 56, 6) format.

Inter-X / InterMask evaluation expects:
  - per-clip array of shape (T, 56, 6) float32
  - axis 1: 56 SMPL-X joints in their canonical order
  - axis 2: 3 axis-angle values for P1 followed by 3 for P2 (concat on last axis)
  - translations stored in joint slot 55, both persons in P1's first-frame reference

Channel reshuffling from our SMPL-X 168-dim layout to Inter-X 56-joint layout
mirrors Inter-X-main/preprocess/1_prepare_data.py.
"""

from __future__ import annotations

import numpy as np

# Our 168-dim SMPL-X channel slices (see CLAUDE.md / smplx_render.py).
SMPLX_ROOT_ORIENT = slice(0, 3)
SMPLX_TRANS = slice(3, 6)
SMPLX_BODY = slice(6, 69)
SMPLX_LHAND = slice(69, 114)
SMPLX_RHAND = slice(114, 159)
SMPLX_JAW = slice(159, 162)
SMPLX_LEYE = slice(162, 165)
SMPLX_REYE = slice(165, 168)


def smplxFlatToInterxJoints(motion168: np.ndarray) -> np.ndarray:
    """(T, 168) SMPL-X axis-angle -> (T, 56, 3) Inter-X joint-ordered axis-angle.

    Inter-X joint order: root_orient(0), body(1..21), jaw(22), leye(23),
    reye(24), lhand(25..39), rhand(40..54), translation(55).
    """
    assert motion168.ndim == 2 and motion168.shape[1] == 168, motion168.shape
    T = motion168.shape[0]
    out = np.zeros((T, 56, 3), dtype=np.float32)

    out[:, 0, :] = motion168[:, SMPLX_ROOT_ORIENT]
    out[:, 1:22, :] = motion168[:, SMPLX_BODY].reshape(T, 21, 3)
    out[:, 22, :] = motion168[:, SMPLX_JAW]
    out[:, 23, :] = motion168[:, SMPLX_LEYE]
    out[:, 24, :] = motion168[:, SMPLX_REYE]
    out[:, 25:40, :] = motion168[:, SMPLX_LHAND].reshape(T, 15, 3)
    out[:, 40:55, :] = motion168[:, SMPLX_RHAND].reshape(T, 15, 3)
    out[:, 55, :] = motion168[:, SMPLX_TRANS]

    return out


def normalizeTranslation(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract P1's first-frame translation from both actors (Inter-X convention)."""
    base = p1[0, 55, :].copy()
    p1 = p1.copy()
    p2 = p2.copy()
    p1[:, 55, :] -= base
    p2[:, 55, :] -= base

    return p1, p2


def pairToInterxFormat(motion_p1: np.ndarray, motion_p2: np.ndarray) -> np.ndarray:
    """(T, 168) + (T, 168) -> (T, 56, 6) Inter-X canonical format.

    Both inputs are in our SMPL-X 168-dim axis-angle layout, in world frame
    (i.e. after denormalize). Output has P1 axis-angle in last-dim slots [0:3]
    and P2 in [3:6], with translations re-anchored to P1's first frame.
    """
    p1 = smplxFlatToInterxJoints(motion_p1)
    p2 = smplxFlatToInterxJoints(motion_p2)
    p1, p2 = normalizeTranslation(p1, p2)

    return np.concatenate([p1, p2], axis=-1).astype(np.float32)


def interxFormatToSmplxFlat(motionInterx: np.ndarray, actor: int = 0) -> np.ndarray:
    """Inverse of pairToInterxFormat for one actor.

    (T, 56, 6) -> (T, 168). Used for sanity checks: encode-then-decode should
    round-trip exactly (translation re-anchor is the only lossy step).
    """
    assert motionInterx.ndim == 3 and motionInterx.shape[1:] == (56, 6), motionInterx.shape
    perActor = motionInterx[..., actor * 3:(actor + 1) * 3]  # (T, 56, 3)
    T = perActor.shape[0]
    out = np.zeros((T, 168), dtype=np.float32)

    out[:, SMPLX_ROOT_ORIENT] = perActor[:, 0, :]
    out[:, SMPLX_BODY] = perActor[:, 1:22, :].reshape(T, -1)
    out[:, SMPLX_JAW] = perActor[:, 22, :]
    out[:, SMPLX_LEYE] = perActor[:, 23, :]
    out[:, SMPLX_REYE] = perActor[:, 24, :]
    out[:, SMPLX_LHAND] = perActor[:, 25:40, :].reshape(T, -1)
    out[:, SMPLX_RHAND] = perActor[:, 40:55, :].reshape(T, -1)
    out[:, SMPLX_TRANS] = perActor[:, 55, :]

    return out
