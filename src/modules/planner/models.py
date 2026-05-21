from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Position3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def __add__(self, other: Position3D) -> Position3D:
        return Position3D(self.x + other.x, self.y + other.y, self.z + other.z)


@dataclass(slots=True)
class PlannedEntity:
    name: str
    object_type: str
    position: Position3D
    rotation: tuple[float, float, float, float] = (0, 0, 0, 1)
    skin: str | None = None
    is_actor: bool = False
    size: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    mass: float = 1.0


@dataclass(slots=True)
class PlannedScene:
    entities: list[PlannedEntity]
    duration: float = 5.0
    actions: list = field(default_factory=list)
