# .\.venv\Scripts\python.exe scripts/extra_motion/fps_study.py aist style100
import csv
import pathlib
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

REPO = pathlib.Path(__file__).resolve().parents[2]
JOINTS60 = REPO / "data" / "extra_motion" / "joints60"
HML_JOINTS = REPO / "data" / "HumanML3D_263" / "new_joints"
OUT_CSV = REPO / "doc" / "notes" / "fps_study.csv"
SOURCE_FPS = 60
TARGET_FPS = 20
STRIDE = SOURCE_FPS // TARGET_FPS
NYQUIST_TARGET = TARGET_FPS / 2
LOWPASS_HZ = 8.0
LOWPASS_ORDER = 4
HML_CLIPS = 400
COLUMNS = (
    "source",
    "fps",
    "clips",
    "hours",
    "share_above_10hz",
    "share_5_to_10hz",
    "phase_0_1_over_step",
    "phase_0_2_over_step",
    "lowpass_over_step",
)


def __band_energy(joints: np.ndarray, fps: int, low: float, high: float) -> tuple[float, float]:
    velocity = np.diff(joints, axis=0) * fps
    power = (np.abs(np.fft.rfft(velocity, axis=0)) ** 2).sum(axis=(1, 2))
    freq = np.fft.rfftfreq(velocity.shape[0], d=1.0 / fps)
    keep = freq > 0
    band = (freq > low) & (freq <= high)
    return float(power[band].sum()), float(power[keep].sum())


def __rms(values: np.ndarray) -> float:
    return float(np.sqrt((values**2).sum(axis=-1).mean()))


def __phase_and_lowpass(joints: np.ndarray, sos: np.ndarray) -> tuple[float, float, float, float]:
    phases = [joints[offset::STRIDE] for offset in range(STRIDE)]
    length = min(len(phase) for phase in phases)
    phases = [phase[:length] for phase in phases]
    step = __rms(np.diff(phases[0], axis=0))
    filtered = sosfiltfilt(sos, joints, axis=0)[::STRIDE][:length]
    return (
        __rms(phases[0] - phases[1]) / step,
        __rms(phases[0] - phases[2]) / step,
        __rms(phases[0] - filtered) / step,
        step,
    )


def __source_row(name: str, files: list[pathlib.Path]) -> tuple:
    sos = np.asarray(butter(LOWPASS_ORDER, LOWPASS_HZ, fs=SOURCE_FPS, output="sos"))
    above, between, total, frames = 0.0, 0.0, 0.0, 0
    ratios = []
    for path in files:
        joints = np.load(path).astype(np.float64)
        high, whole = __band_energy(joints, SOURCE_FPS, NYQUIST_TARGET, SOURCE_FPS)
        mid, _ = __band_energy(joints, SOURCE_FPS, NYQUIST_TARGET / 2, NYQUIST_TARGET)
        above, between, total = above + high, between + mid, total + whole
        frames += joints.shape[0]
        ratios.append(__phase_and_lowpass(joints, sos)[:3])
    mean_ratios = np.mean(ratios, axis=0)
    return (
        name,
        SOURCE_FPS,
        len(files),
        round(frames / SOURCE_FPS / 3600, 2),
        round(above / total, 4),
        round(between / total, 4),
        round(float(mean_ratios[0]), 4),
        round(float(mean_ratios[1]), 4),
        round(float(mean_ratios[2]), 4),
    )


def __hml_row() -> tuple:
    files = sorted(HML_JOINTS.glob("*.npy"))[:HML_CLIPS]
    if not files:
        raise FileNotFoundError(f"no joints under {HML_JOINTS}")
    between, total, frames = 0.0, 0.0, 0
    for path in files:
        joints = np.load(path).astype(np.float64)[:, :22]
        mid, whole = __band_energy(joints, TARGET_FPS, NYQUIST_TARGET / 2, NYQUIST_TARGET)
        between, total, frames = between + mid, total + whole, frames + joints.shape[0]
    return (
        "humanml3d_reference",
        TARGET_FPS,
        len(files),
        round(frames / TARGET_FPS / 3600, 2),
        "",
        round(between / total, 4),
        "",
        "",
        "",
    )


def main() -> None:
    names = sys.argv[1:]
    if not names:
        raise SystemExit("give at least one source folder name under data/extra_motion/joints60")
    rows = [__hml_row()]
    for name in names:
        files = sorted((JOINTS60 / name).glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"no 60 fps joints under {JOINTS60 / name}")
        rows.append(__source_row(name, files))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    for row in rows:
        print(dict(zip(COLUMNS, row, strict=True)))


if __name__ == "__main__":
    main()
