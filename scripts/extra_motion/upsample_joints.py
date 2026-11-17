# .\.venv\Scripts\python.exe scripts/extra_motion/upsample_joints.py motionx_official 30
import pathlib
import sys

import numpy as np
from scipy.interpolate import CubicSpline

REPO = pathlib.Path(__file__).resolve().parents[2]
EXTRA = REPO / "data" / "extra_motion"
TARGET_FPS = 60
MIN_SOURCE_FRAMES = 4


def __cubic_spline(joints: np.ndarray, source_fps: int) -> np.ndarray:
    times = np.arange(joints.shape[0]) / source_fps
    targets = np.arange(round(times[-1] * TARGET_FPS) + 1) / TARGET_FPS
    spline = CubicSpline(times, joints.astype(np.float64), axis=0)
    return spline(np.minimum(targets, times[-1])).astype(np.float32)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: upsample_joints.py <source> <source_fps>")
    source, source_fps = sys.argv[1], int(sys.argv[2])
    if TARGET_FPS % source_fps != 0:
        raise ValueError(f"{source_fps} fps does not divide {TARGET_FPS} fps")
    in_dir = EXTRA / f"joints{source_fps}" / source
    if not in_dir.is_dir():
        raise FileNotFoundError(f"no joints folder {in_dir}")
    out_dir = EXTRA / f"joints{TARGET_FPS}" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped, seconds = 0, 0, 0.0
    for path in sorted(in_dir.glob("*.npy")):
        joints = np.load(path)
        if not np.isfinite(joints).all():
            raise ValueError(f"{path}: non-finite joints")
        if joints.shape[0] < MIN_SOURCE_FRAMES:
            skipped += 1
            continue
        upsampled = __cubic_spline(joints, source_fps)
        np.save(out_dir / path.name, upsampled)
        written += 1
        seconds += (upsampled.shape[0] - 1) / TARGET_FPS
    print(f"{source}: written {written}  skipped {skipped}  hours {seconds / 3600:.2f}")


if __name__ == "__main__":
    main()
