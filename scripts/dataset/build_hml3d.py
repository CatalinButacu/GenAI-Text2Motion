# .\.venv\Scripts\python.exe scripts/dataset/build_hml3d.py all --fps 60 [--amass-raw DIR] [--work DIR]
import argparse
import pathlib

from text2motion.dataset.contracts import DatasetPaths
from text2motion.dataset.pipeline import request_from_paths, run_pipeline
from text2motion.motion.contracts import PreparationStage

REPO = pathlib.Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[stage.value for stage in PreparationStage])
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--amass-raw", type=pathlib.Path, default=REPO / "data" / "amass")
    parser.add_argument("--work", type=pathlib.Path, default=None)
    parser.add_argument("--no-fill", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    paths = DatasetPaths(REPO, arguments.amass_raw.resolve(), work=arguments.work)
    request = request_from_paths(paths, arguments.fps, fill=not arguments.no_fill)
    run_pipeline(request, PreparationStage(arguments.stage))


if __name__ == "__main__":
    main()
