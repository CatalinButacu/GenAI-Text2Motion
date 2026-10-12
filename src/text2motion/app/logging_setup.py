from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RUN_LOG_NAME = "run.log"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configure_logging(level: str = "INFO") -> None:
    if level not in LOG_LEVELS:
        raise ValueError(f"unknown log level {level!r} (expected one of {LOG_LEVELS})")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(console)
    root.setLevel(level)


def ensure_logging_configured() -> None:
    if not logging.getLogger().handlers:
        configure_logging()


def add_run_log_file(run_dir: Path) -> None:
    run_log = logging.FileHandler(Path(run_dir) / RUN_LOG_NAME, encoding="utf-8")
    run_log.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logging.getLogger().addHandler(run_log)
