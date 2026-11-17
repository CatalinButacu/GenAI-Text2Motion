# .\.venv\Scripts\python.exe scripts/dataset/build_hml3d.py all --fps 60
import argparse
import csv
import pathlib

from text2motion.dataset.clock import stored_fps_table
from text2motion.dataset.contracts import AMASS_LABEL_FPS, DatasetPaths
from text2motion.dataset.preparation import (
    Hml3dPreparationLayout,
    build_features,
    build_joint_clips,
    preparation_request,
    run_hml3d_preparation,
)
from text2motion.motion.contracts import PreparationStage

REPO = pathlib.Path(__file__).resolve().parents[2]
PATHS = DatasetPaths(REPO, pathlib.Path(r"D:\Facultate\dissertation\data\amass"))
CLOCK_CSV = PATHS.notes / "amass_clock_check.csv"
STAGES = ("clock", "joints", "features", "all")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--fps", type=int, required=True)
    return parser.parse_args()


def write_clock_table(fps: int) -> None:
    request = preparation_request(PATHS, fps)
    run_hml3d_preparation(request, PreparationStage.AMASS)
    pose_root = Hml3dPreparationLayout(request.out_dir).pose_data
    rows = [
        row.as_record()
        for row in stored_fps_table(PATHS.hml3d_official, pose_root, AMASS_LABEL_FPS // fps)
    ]
    with open(CLOCK_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    statuses = [str(row["status"]) for row in rows]
    print("files", len(rows), {status: statuses.count(status) for status in sorted(set(statuses))})


def main() -> None:
    arguments = parse_arguments()
    stage, fps = arguments.stage, arguments.fps
    if stage in ("clock", "all"):
        write_clock_table(fps)
    if stage in ("joints", "all"):
        build_joint_clips(PATHS, CLOCK_CSV, fps)
    if stage in ("features", "all"):
        build_features(PATHS, fps)


if __name__ == "__main__":
    main()
