# .\.venv\Scripts\python.exe scripts/dataset/scan_clips.py <folder-of-npy> <log.csv> [--min-frames 40]
import argparse
import pathlib

from text2motion.dataset.core.scan import scan_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=pathlib.Path)
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--min-frames", type=int, default=40)
    arguments = parser.parse_args()
    print("clips by action", scan_folder(arguments.folder, arguments.log, arguments.min_frames))


if __name__ == "__main__":
    main()
