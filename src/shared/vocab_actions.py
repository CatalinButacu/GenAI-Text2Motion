from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cache
from pathlib import Path

import yaml

VOCAB_DIR = Path(__file__).parents[2] / "data" / "vocabulary"


class ActionCategory(Enum):
    LOCOMOTION = auto()
    MANIPULATION = auto()
    INTERACTION = auto()
    GESTURE = auto()
    POSE = auto()
    PHYSICS = auto()
    EMOTION = auto()
    ATHLETICS = auto()


@dataclass(slots=True)
class ActionDefinition:
    name: str
    category: ActionCategory
    keywords: list[str] = field(default_factory=list)
    requiresTarget: bool = False
    motionClip: str | None = None


@cache
def loadActions() -> dict[str, ActionDefinition]:
    with open(VOCAB_DIR / "actions.yaml", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    return {
        name: ActionDefinition(
            name=name,
            category=ActionCategory[d["category"]],
            keywords=[str(k) for k in d.get("keywords", [name])],
            requiresTarget=bool(d.get("requires_target", False)),
            motionClip=d.get("motion_clip") or None,
        )
        for name, d in raw.items()
    }


ACTIONS: dict[str, ActionDefinition] = loadActions()
