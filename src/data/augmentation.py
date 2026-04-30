from __future__ import annotations

import logging
import re

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from .aug_quality import detectTpose, qualityFilter
from .aug_slerp import slerpResample

log = logging.getLogger(__name__)


# ---- Mirror-flip augmentation (biomechanically correct L<->R swap) ----

# SMPL-X 168-dim pose layout:
#   0:3      root orient (axis-angle)
#   3:6      root translation
#   6:69     body pose (21 joints * 3 axes), joints 1..21 indexed from 0
#   69:114   left hand pose (15 joints * 3)
#   114:159  right hand pose (15 joints * 3)
#   159:168  jaw + eyes (always zero in AMASS -- dead channels)
#
# Body joint 0-indexed (within the 21-joint body pose block):
#   0=L_hip  1=R_hip   3=L_knee  4=R_knee   6=L_ankle 7=R_ankle
#   9=L_foot 10=R_foot  12=L_col 13=R_col   15=L_sho  16=R_sho
#   17=L_elb 18=R_elb   19=L_wri 20=R_wri
BODY_LR_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16), (17, 18), (19, 20),
)


def mirrorRotvecInplace(m: np.ndarray, start: int, end: int) -> None:
    """Negate y,z components of every 3D rotvec in channels[start:end].

    Under mirror M=diag(-1,1,1): rotvec (rx,ry,rz) -> (rx,-ry,-rz).
    """
    m[:, start + 1:end:3] *= -1.0
    m[:, start + 2:end:3] *= -1.0


def mirrorFlip(motion: np.ndarray) -> np.ndarray:
    """Mirror the pose left<->right across the YZ plane (x <-> -x).

    Produces a biomechanically valid mirrored sequence: requires matching
    text swap (done by caller). No-op if the feature vector is smaller than
    the full SMPL-X 168 layout.
    """
    if motion.ndim != 2 or motion.shape[1] < 159 or motion.shape[0] == 0:
        return motion

    m = motion.copy()

    mirrorRotvecInplace(m, 0, 3)     # root orient
    m[:, 3] *= -1.0                      # translation x -> -x

    bodyStart = 6
    mirrorRotvecInplace(m, bodyStart, bodyStart + 63)

    for a, b in BODY_LR_PAIRS:
        ia, ib = bodyStart + 3 * a, bodyStart + 3 * b
        m[:, [ia, ia + 1, ia + 2, ib, ib + 1, ib + 2]] = (
            m[:, [ib, ib + 1, ib + 2, ia, ia + 1, ia + 2]]
        )

    l_start, r_start, hand_len = 69, 114, 45
    leftBlock = m[:, l_start:l_start + hand_len].copy()
    m[:, l_start:l_start + hand_len] = m[:, r_start:r_start + hand_len]
    m[:, r_start:r_start + hand_len] = leftBlock
    mirrorRotvecInplace(m, l_start, l_start + hand_len)
    mirrorRotvecInplace(m, r_start, r_start + hand_len)

    return m.astype(np.float32)


LR_WORD_RE = re.compile(r"\b(left|right|Left|Right|LEFT|RIGHT)\b")
LR_SWAP = {"left": "right", "right": "left", "Left": "Right", "Right": "Left",
            "LEFT": "RIGHT", "RIGHT": "LEFT"}


def mirrorFlipText(text: str) -> str:
    """Swap left<->right tokens so text matches a mirror-flipped motion."""
    return LR_WORD_RE.sub(lambda m: LR_SWAP[m.group(0)], text)


def canonicalizeRoot(motion: np.ndarray) -> np.ndarray:
    """Zero the frame-0 XY translation AND remove frame-0 yaw.

    Z-up convention: channels 3,4 = (tx,ty), channel 5 = tz (height).
    Step 1: subtract frame-0 (tx,ty) from all frames so each clip starts at origin
            (Z left untouched — feet-above-floor is a physically meaningful absolute).
    Step 2: extract the yaw (rotation around vertical axis Z) from the frame-0 root
            orientation (channels 0:3 axis-angle), and apply its inverse to every
            frame's root orientation AND root translation. Removes spurious global
            heading variance: "walks forward facing east" and "walks forward facing
            north" collapse to the same feature trajectory, freeing codebook capacity
            for actual motion content.
    """
    if motion.ndim != 2 or motion.shape[1] < 6 or motion.shape[0] == 0:
        return motion

    m = motion.copy()
    m[:, 3:5] -= m[0:1, 3:5]

    rot0 = R.from_rotvec(m[0, 0:3])
    yaw0 = float(rot0.as_euler("ZYX")[0])

    if abs(yaw0) < 1e-4:
        return m

    yawInv = R.from_euler("Z", -yaw0)
    rots = R.from_rotvec(m[:, 0:3])
    m[:, 0:3] = (yawInv * rots).as_rotvec().astype(m.dtype)
    m[:, 3:6] = yawInv.apply(m[:, 3:6]).astype(m.dtype)

    return m


def resampleToFps(motion: np.ndarray, srcFps: float, tgtFps: float = 30.0) -> np.ndarray:
    """Time-resample a motion clip from srcFps to tgtFps.

    SLERP is used for full SMPL-X (168-dim) so axis-angle channels interpolate
    on the rotation manifold. Linear interp is used for non-rotation tensors.
    Refuses to resample when input is degenerate (NaN, <=1 frame, fps<=1 Hz),
    returning the input unchanged so the caller's quality filter rejects it.
    """
    if not np.isfinite(srcFps) or srcFps <= 1.0:
        return motion

    if not np.isfinite(tgtFps) or tgtFps <= 1.0:
        return motion

    if abs(srcFps - tgtFps) < 0.5:
        return motion

    T_src = motion.shape[0]

    if T_src <= 1:
        return motion

    if not np.isfinite(motion).all():
        return motion

    duration = T_src / srcFps
    T_tgt = max(1, int(round(duration * tgtFps)))
    srcT = np.linspace(0, duration, T_src)
    tgtT = np.linspace(0, duration, T_tgt)

    if motion.shape[1] == 168:
        return slerpResample(motion, srcT, tgtT)

    f = interp1d(srcT, motion, axis=0, assume_sorted=True)

    return f(tgtT).astype(motion.dtype)


def temporalCrop(
    motion: np.ndarray, maxLength: int, rng: np.random.Generator | None = None,
) -> np.ndarray:
    T = motion.shape[0]

    if T <= maxLength:
        return motion

    rng = rng or np.random.default_rng()
    start = rng.integers(0, T - maxLength + 1)

    return motion[start:start + maxLength]


def speedPerturbation(
    motion: np.ndarray, rng: np.random.Generator | None = None,
    speedRange: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    factor = rng.uniform(*speedRange)

    if abs(factor - 1.0) < 0.02:
        return motion

    T = motion.shape[0]
    T_new = max(1, int(round(T / factor)))
    srcT = np.linspace(0, 1, T)
    tgtT = np.linspace(0, 1, T_new)

    if motion.shape[1] == 168:
        return slerpResample(motion, srcT, tgtT)

    f = interp1d(srcT, motion, axis=0, assume_sorted=True)

    return f(tgtT).astype(motion.dtype)


def addNoise(
    motion: np.ndarray, sigma: float = 0.002, rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, motion.shape).astype(motion.dtype)

    if motion.shape[1] >= 6:
        noise[:, 3:6] = 0.0

    return motion + noise


class AugmentationPipeline:
    def __init__(
        self, *, temporalCropEnabled: bool = True, speedPerturbEnabled: bool = True,
        noiseEnabled: bool = True, noiseSigma: float = 0.002,
        speedRange: tuple[float, float] = (0.8, 1.2), maxLength: int = 200,
        speedProb: float = 0.5, noiseProb: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.temporalCropEnabled = temporalCropEnabled
        self.speedPerturbEnabled = speedPerturbEnabled
        self.noiseEnabled = noiseEnabled
        self.noiseSigma = noiseSigma
        self.speedRange = speedRange
        self.maxLength = maxLength
        self.speedProb = speedProb
        self.noiseProb = noiseProb
        self.rng = np.random.default_rng(seed)

    def __call__(self, motion: np.ndarray) -> np.ndarray:
        if self.speedPerturbEnabled and self.rng.random() < self.speedProb:
            motion = speedPerturbation(motion, self.rng, self.speedRange)

        if self.temporalCropEnabled:
            motion = temporalCrop(motion, self.maxLength, self.rng)

        if self.noiseEnabled and self.rng.random() < self.noiseProb:
            motion = addNoise(motion, self.noiseSigma, self.rng)

        return motion

    def invoke(self, motion: np.ndarray) -> np.ndarray:
        return self(motion)
