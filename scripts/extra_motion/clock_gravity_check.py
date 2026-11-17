# .\.venv\Scripts\python.exe scripts/extra_motion/clock_gravity_check.py 60 "DFaust=data/HumanML3D_60fps/pose_data/DFaust/**/*.npy" "GRAB=data/extra_motion/joints60/amass_extra/GRAB__*.npy"
import csv
import glob
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_CSV = REPO / "doc" / "notes" / "amass_clock_gravity.csv"
GRAVITY_M_PER_S2 = 9.81
FEET = (7, 8, 10, 11)
UP_AXIS = 1
FLOOR_PERCENTILE = 2
FLIGHT_CLEARANCE_M = 0.06
MIN_FLIGHT_SECONDS = 0.1


def __flight_runs(airborne: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], airborne.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1).tolist()
    ends = np.flatnonzero(edges == -1).tolist()
    return list(zip(starts, ends, strict=True))


def __flight_accelerations(path: str, fps: int) -> list[float]:
    joints = np.load(path)
    heights = joints[..., UP_AXIS]
    feet = heights[:, FEET].min(axis=1)
    floor = np.percentile(feet, FLOOR_PERCENTILE)
    centre = heights.mean(axis=1)
    accelerations = []
    for start, end in __flight_runs(feet - floor > FLIGHT_CLEARANCE_M):
        if end - start < MIN_FLIGHT_SECONDS * fps:
            continue
        times = np.arange(end - start) / fps
        accelerations.append(2.0 * float(np.polyfit(times, centre[start:end], 2)[0]))
    return accelerations


def __clock_row(label: str, pattern: str, fps: int) -> dict[str, object]:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"{label}: no file matches {pattern}")
    values = np.array([a for path in files for a in __flight_accelerations(path, fps)])
    median = float(np.median(values)) if values.size else float("nan")
    return {
        "source": label,
        "files": len(files),
        "flights": int(values.size),
        "median_m_per_s2": round(median, 2),
        "q25": round(float(np.percentile(values, 25)), 2) if values.size else "",
        "q75": round(float(np.percentile(values, 75)), 2) if values.size else "",
        "speed_factor": round(float(np.sqrt(-median / GRAVITY_M_PER_S2)), 2)
        if values.size and median < 0
        else "",
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: clock_gravity_check.py <fps> <label=glob> [<label=glob> ...]")
    fps = int(sys.argv[1])
    pairs = [argument.partition("=") for argument in sys.argv[2:]]
    rows = [__clock_row(label, pattern, fps) for label, _, pattern in pairs]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
