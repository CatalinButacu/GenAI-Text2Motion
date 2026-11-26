# .\.venv\Scripts\python.exe scripts/dataset/check_quality.py quality --fps 60 amass_extra motionx_official
import argparse
import csv
import glob
import pathlib
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from text2motion.dataset.contracts import (
    GRAVITY_M_PER_S2,
    MIN_CLIP_SECONDS,
    DatasetPaths,
    NewtonLimits,
    QualityLimits,
    subset_of,
)
from text2motion.dataset.core.encoding import feature_joints
from text2motion.dataset.core.quality import (
    QUALITY_METRICS,
    ClipVerdict,
    fit_quality_forest,
    flight_accelerations,
    in_step0,
    newton_violations,
    quality_inliers,
    quality_metrics,
    summary_table,
)
from text2motion.motion.storage import base_clip_ids

REPO = pathlib.Path(__file__).resolve().parents[2]
PATHS = DatasetPaths(REPO, REPO / "data" / "amass")
NEWTON = NewtonLimits()
QUALITY = QualityLimits()
HML3D_TRAIN, HML3D_VAL = "hml3d_train", "hml3d_val"
FLIGHT_CLEARANCE_M = 0.06
MIN_FLIGHT_SECONDS = 0.1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    quality = commands.add_parser("quality")
    quality.add_argument("--fps", type=int, required=True)
    quality.add_argument("sources", nargs="+")
    gravity = commands.add_parser("gravity")
    gravity.add_argument("--fps", type=int, required=True)
    gravity.add_argument("patterns", nargs="+", help="label=glob of joint files")
    return parser.parse_args()


def hml3d_dir(fps: int) -> pathlib.Path:
    return PATHS.hml3d_official if fps == 20 else PATHS.hml3d_at(fps)


def source_files(source: str, fps: int) -> list[pathlib.Path]:
    if source in (HML3D_TRAIN, HML3D_VAL):
        root = hml3d_dir(fps)
        split = "train" if source == HML3D_TRAIN else "val"
        paths = [root / "new_joint_vecs" / f"{n}.npy" for n in base_clip_ids(root, split)]
        return [path for path in paths if path.is_file()]
    return sorted(PATHS.features(fps, source).glob("*.npy"))


def measure(source: str, fps: int) -> list[dict[str, object]]:
    rows = []
    for path in tqdm(source_files(source, fps), desc=source):
        features = np.load(path).astype(np.float32)
        if features.shape[0] < MIN_CLIP_SECONDS * fps:
            continue
        newton = newton_violations(feature_joints(features), fps, NEWTON)
        row: dict[str, object] = {
            "source": source,
            "subset": subset_of(path.stem, source),
            "name": path.stem,
            "seconds": features.shape[0] / fps,
            "hover_share": round(newton.hover_share, 4),
            "unbalanced_share": round(newton.unbalanced_share, 4),
            "violation_share": round(newton.violation_share, 4),
        }
        row.update(zip(QUALITY_METRICS, quality_metrics(features, fps, QUALITY), strict=True))
        rows.append(row)
    return rows


def metric_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    return np.array([[float(str(row[key])) for key in QUALITY_METRICS] for row in rows])


def to_verdict(row: dict[str, object]) -> ClipVerdict:
    return ClipVerdict(
        source=str(row["source"]),
        subset=str(row["subset"]),
        name=str(row["name"]),
        seconds=float(str(row["seconds"])),
        violation_share=float(str(row["violation_share"])),
        inlier=bool(row["inlier"]),
    )


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_quality_report(fps: int, sources: list[str]) -> None:
    train = measure(HML3D_TRAIN, fps)
    rows = train + measure(HML3D_VAL, fps)
    for source in sources:
        rows += measure(source, fps)
    forest = fit_quality_forest(metric_matrix(train), QUALITY)
    for row, inlier in zip(rows, quality_inliers(forest, metric_matrix(rows)), strict=True):
        verdict = to_verdict({**row, "inlier": bool(inlier)})
        row["inlier"] = verdict.inlier
        row["kept"] = verdict.kept(NEWTON)
        row["in_step0"] = in_step0(verdict.source, verdict.subset)
    write_csv(PATHS.extra / f"quality_clips_{fps}fps.csv", rows)
    table = summary_table([to_verdict(row) for row in rows], NEWTON)
    write_csv(PATHS.notes / f"quality_{fps}fps.csv", table)
    for line in table:
        print(line)
    print("limits", asdict(NEWTON), asdict(QUALITY))


def gravity_row(label: str, pattern: str, fps: int) -> dict[str, object]:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"{label}: no file matches {pattern}")
    values = np.array(
        [
            value
            for path in files
            for value in flight_accelerations(
                np.load(path), fps, FLIGHT_CLEARANCE_M, MIN_FLIGHT_SECONDS
            )
        ]
    )
    median = float(np.median(values)) if values.size else float("nan")
    factor = np.sqrt(-median / GRAVITY_M_PER_S2) if values.size and median < 0 else float("nan")
    return {
        "source": label,
        "files": len(files),
        "flights": int(values.size),
        "median_m_per_s2": round(median, 2),
        "speed_factor": round(float(factor), 2),
    }


def write_gravity_report(fps: int, patterns: list[str]) -> None:
    pairs = [argument.partition("=") for argument in patterns]
    rows = [gravity_row(label, pattern, fps) for label, _, pattern in pairs]
    write_csv(PATHS.notes / "amass_clock_gravity.csv", rows)
    for row in rows:
        print(row)


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "quality":
        write_quality_report(arguments.fps, arguments.sources)
        return
    write_gravity_report(arguments.fps, arguments.patterns)


if __name__ == "__main__":
    main()
