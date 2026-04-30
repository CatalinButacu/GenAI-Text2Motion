from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

ROTATION_SLICES = [slice(0, 3)] + [
    slice(6 + i * 3, 6 + (i + 1) * 3) for i in range(54)  # body+hands+jaw+eyes channels
]
TRANSLATION_SLICE = slice(3, 6)  # root translation channels (linear interpolation)


def slerpResample(motion: np.ndarray, srcTimes: np.ndarray, tgtTimes: np.ndarray) -> np.ndarray:
    D = motion.shape[1]
    out = np.empty((len(tgtTimes), D), dtype=motion.dtype)
    fTrans = interp1d(srcTimes, motion[:, TRANSLATION_SLICE], axis=0, assume_sorted=True)
    out[:, TRANSLATION_SLICE] = fTrans(tgtTimes)

    for sl in ROTATION_SLICES:
        aa = motion[:, sl]
        try:
            rots = Rotation.from_rotvec(aa)
            slerp = Slerp(srcTimes, rots)
            out[:, sl] = slerp(tgtTimes).as_rotvec().astype(motion.dtype)
        except ValueError:
            f = interp1d(srcTimes, aa, axis=0, assume_sorted=True)
            out[:, sl] = f(tgtTimes)

    return out
