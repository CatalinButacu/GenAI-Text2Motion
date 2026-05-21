from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

ROTATION_SLICES = [slice(0, 3)] + [
    slice(6 + i * 3, 6 + (i + 1) * 3) for i in range(54)  # body+hands+jaw+eyes channels
]
TRANSLATION_SLICE = slice(3, 6)  # root translation channels (linear interpolation)


def slerp_resample(motion: np.ndarray, src_times: np.ndarray, tgt_times: np.ndarray) -> np.ndarray:
    D = motion.shape[1]
    out = np.empty((len(tgt_times), D), dtype=motion.dtype)
    f_trans = interp1d(src_times, motion[:, TRANSLATION_SLICE], axis=0, assume_sorted=True)
    out[:, TRANSLATION_SLICE] = f_trans(tgt_times)

    for sl in ROTATION_SLICES:
        aa = motion[:, sl]
        try:
            rots = Rotation.from_rotvec(aa)
            slerp = Slerp(src_times, rots)
            out[:, sl] = slerp(tgt_times).as_rotvec().astype(motion.dtype)
        except ValueError:
            f = interp1d(src_times, aa, axis=0, assume_sorted=True)
            out[:, sl] = f(tgt_times)

    return out
