from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROTATION_SLICES = [slice(0, 3)] + [
    slice(6 + i * 3, 6 + (i + 1) * 3) for i in range(54)
]
TRANSLATION_SLICE = slice(3, 6)


def slerp_blend_frames(a: np.ndarray, b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Per-frame SLERP blend of two SMPL-X 168-dim pose arrays weighted by alpha."""
    N = a.shape[0]
    result = np.empty_like(a)
    w = alpha[:, None]
    result[:, TRANSLATION_SLICE] = (
        a[:, TRANSLATION_SLICE] * (1 - w) + b[:, TRANSLATION_SLICE] * w
    )

    for sl in ROTATION_SLICES:
        for i in range(N):
            t = float(alpha[i])

            if t <= 0.0:
                result[i, sl] = a[i, sl]
                continue

            if t >= 1.0:
                result[i, sl] = b[i, sl]
                continue

            try:
                ra = Rotation.from_rotvec(a[i, sl])
                rb = Rotation.from_rotvec(b[i, sl])
                result[i, sl] = (
                    Slerp([0.0, 1.0], Rotation.concatenate([ra, rb]))([t])
                    .as_rotvec()[0]
                    .astype(a.dtype)
                )
            except ValueError:
                result[i, sl] = a[i, sl] * (1 - t) + b[i, sl] * t

    return result
