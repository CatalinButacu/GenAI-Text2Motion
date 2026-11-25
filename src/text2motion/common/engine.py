from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

type EpochGuard = Callable[[int], AbstractContextManager[None]]


############################################################
#       CleanCrossEntropyProgress
############################################################
@dataclass(frozen=True)
class CleanCrossEntropyProgress:
    best: float
    stale: int

    def observe(self, value: float) -> CleanCrossEntropyProgress:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid clean validation CE: {value}")
        if value < self.best:
            return CleanCrossEntropyProgress(best=value, stale=0)
        return CleanCrossEntropyProgress(best=self.best, stale=self.stale + 1)


def read_clean_ce_history(metrics_path: Path, completed_epochs: int) -> dict[int, float]:
    history: dict[int, float] = {}
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "val_ce" not in record:
                continue
            epoch = record["epoch"]
            if type(epoch) is not int or epoch < 1:
                raise ValueError(f"invalid clean-CE epoch in {metrics_path}: {epoch}")
            if epoch > completed_epochs:
                continue
            if epoch in history:
                logger.warning(
                    f"duplicate clean CE for epoch {epoch} in {metrics_path}; using the latest"
                )
            history[epoch] = float(record["val_ce"])
    return history


def restore_clean_ce(
    state: dict, completed_epochs: int, fresh_selection: bool, val_ce_patience: int
) -> CleanCrossEntropyProgress:
    if fresh_selection or not val_ce_patience:
        return CleanCrossEntropyProgress(best=float("inf"), stale=0)
    saved = state.get("clean_ce_progress")
    if saved is not None:
        progress = CleanCrossEntropyProgress(best=float(saved["best"]), stale=saved["stale"])
    else:
        recorded_dir = state.get("run_dir")
        if recorded_dir is None:
            raise ValueError(
                "resume lacks validation-CE patience and a recorded run directory; "
                "use --fresh_selection only if intentionally resetting selection"
            )
        metrics_path = Path(recorded_dir) / "metrics.jsonl"
        history = read_clean_ce_history(metrics_path, completed_epochs)
        expected = set(range(1, completed_epochs + 1))
        if set(history) != expected:
            raise ValueError(
                f"cannot restore validation-CE patience: {metrics_path} lacks completed epochs "
                f"{sorted(expected - set(history))}; use --fresh_selection only for an intentional reset"
            )
        progress = CleanCrossEntropyProgress(best=float("inf"), stale=0)
        for epoch in range(1, completed_epochs + 1):
            progress = progress.observe(history[epoch])
        logger.info(f"restored validation-CE patience from completed epochs in {metrics_path}")
    if (
        not math.isfinite(progress.best)
        or progress.best < 0
        or type(progress.stale) is not int
        or not 0 <= progress.stale < completed_epochs
    ):
        raise ValueError("invalid saved validation-CE patience state")
    logger.info(f"validation-CE resume: best {progress.best:.6f}, stale {progress.stale} epochs")
    return progress


def run_epochs(
    start_epoch: int,
    epochs: int,
    train_epoch: Callable[[int], dict[str, float]],
    after_epoch: Callable[[int, dict[str, float]], str],
    guard: EpochGuard,
) -> None:
    for epoch in range(start_epoch, epochs):
        with guard(epoch):
            means = train_epoch(epoch)
        reason = after_epoch(epoch, means)
        if reason:
            logger.info(reason)
            break
