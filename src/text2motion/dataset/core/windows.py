from __future__ import annotations

import numpy as np


class CropError(ValueError):
    pass


def cut_mirrored_window(
    data: np.ndarray, start: int, end: int, trim: int, whole: bool = False
) -> np.ndarray:
    if whole:
        return data
    data = data[trim:]
    if start < 0 or end <= start:
        raise CropError("bad crop bounds")
    clip = data[start:end].copy()
    clip[..., 0] *= -1
    if clip.shape[0] < 2:
        raise CropError("fewer than 2 frames")
    return clip
