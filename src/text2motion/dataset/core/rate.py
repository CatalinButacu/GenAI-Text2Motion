from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, resample_poly, sosfiltfilt

LOWPASS_ORDER = 4
WHOLE_FRAME_TOLERANCE = 1e-6


def cubic_resample(joints: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    def target_frame_count(source_frames: int, source_fps: float, target_fps: float) -> int:
        span = (source_frames - 1) / source_fps * target_fps
        nearest = round(span)
        whole = abs(span - nearest) < WHOLE_FRAME_TOLERANCE
        return (nearest if whole else int(np.floor(span))) + 1

    if joints.shape[0] < 2:
        raise ValueError(f"a cubic spline needs 2 or more frames, got {joints.shape[0]}")
    times = np.arange(joints.shape[0]) / source_fps
    count = target_frame_count(joints.shape[0], source_fps, target_fps)
    targets = np.minimum(np.arange(count) / target_fps, times[-1])
    curve = CubicSpline(times, joints.astype(np.float64), axis=0)
    return curve(targets).astype(np.float32)


def lowpass(joints: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    sos = np.asarray(butter(LOWPASS_ORDER, cutoff_hz, fs=fps, output="sos"))
    return sosfiltfilt(sos, joints.astype(np.float64), axis=0)


def downsample(joints: np.ndarray, source_fps: int, target_fps: int) -> np.ndarray:
    if source_fps == target_fps:
        return joints
    if source_fps % target_fps == 0:
        return joints[:: source_fps // target_fps]
    return resample_poly(joints, target_fps, source_fps, axis=0, padtype="line")
