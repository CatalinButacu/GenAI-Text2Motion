from __future__ import annotations

from pathlib import Path

from text2motion.dataset.splits.plan import SplitPlan
from text2motion.motion.contracts import Split
from text2motion.motion.storage import base_clip_ids


class TrainView:
    def __init__(self, plan: SplitPlan) -> None:
        self._names = plan.training_names()

    def allows(self, name: str) -> bool:
        return name in self._names

    def require(self, name: str) -> str:
        if name not in self._names:
            raise PermissionError(
                f"clip {name} is not a training clip; training reads train clips "
                "from unprotected sources only"
            )
        return name

    def names(self) -> frozenset[str]:
        return self._names


ALLOWED_TRAIN_FILE = "train_allowed.txt"


def split_clip_ids(root: Path, split: Split | str) -> list[str]:
    listed = base_clip_ids(root, split)
    if Split(split) is not Split.TRAIN:
        return listed
    allowed_path = Path(root) / ALLOWED_TRAIN_FILE
    if not allowed_path.is_file():
        raise FileNotFoundError(
            f"{allowed_path} missing; training reads only the allowed list (rebuild the features)"
        )
    allowed = {
        line.strip()
        for line in allowed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return [name for name in listed if name in allowed]
