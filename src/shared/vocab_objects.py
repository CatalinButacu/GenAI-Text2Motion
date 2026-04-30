from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cache
from pathlib import Path

import yaml

VOCAB_DIR = Path(__file__).parents[2] / "data" / "vocabulary"


class ObjectCategory(Enum):
    # Physical-role categories -- maps directly to simulation behaviour:
    PRIMITIVE = auto()  #   PRIMITIVE  raw geometric shape, no semantic meaning
    PROP = auto()  #   PROP       static/environment object (immovable scene dressing)
    OBJECT = auto()  #   OBJECT     interactive everyday object (graspable / throwable / dynamic)
    HUMANOID = auto()  #   HUMANOID   human or character entity (URDF + SMPL-X driven)


@dataclass(slots=True)
class ObjectDefinition:
    name: str
    category: ObjectCategory
    keywords: list[str] = field(default_factory=list)
    defaultShape: str = "box"
    defaultSize: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    defaultMass: float = 1.0
    canBeGrasped: bool = True
    meshPrompt: str | None = None
    urdfPath: str | None = None


@cache
def loadObjects() -> dict[str, ObjectDefinition]:
    with open(VOCAB_DIR / "objects.yaml", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    return {
        name: ObjectDefinition(
            name=name,
            category=ObjectCategory[d["category"]],
            keywords=[str(k) for k in d.get("keywords", [name])],
            defaultShape=str(d.get("default_shape", "box")),
            defaultSize=[float(x) for x in d.get("default_size", [0.1, 0.1, 0.1])],
            defaultMass=float(d.get("default_mass", 1.0)),
            canBeGrasped=bool(d.get("can_be_grasped", True)),
            meshPrompt=d.get("mesh_prompt") or None,
            urdfPath=d.get("urdf_path") or None,
        )
        for name, d in raw.items()
    }


OBJECTS: dict[str, ObjectDefinition] = loadObjects()
