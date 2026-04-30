from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedAction:
    actionType: str
    actor: str = ""
    target: str = ""
    rawText: str = ""
    order: int = 0
    duration: float | None = None
    modifier: str = ""


@dataclass(slots=True)
class ParsedEntity:
    name: str
    objectType: str
    isActor: bool = False
    skin: str | None = None
    actions: list[ParsedAction] = field(default_factory=list)


@dataclass(slots=True)
class SpatialRelation:
    subject: str
    predicate: str
    relation: str
    object: str


@dataclass(slots=True)
class ParsedScene:
    prompt: str = ""
    entities: list[ParsedEntity] = field(default_factory=list)
    spatialRelations: list[SpatialRelation] = field(default_factory=list)
    duration: float = 5.0
    durationExplicit: bool = False  # True when prompt contained an explicit duration

    @property
    def actions(self) -> list[ParsedAction]:
        """Flat view of all entity actions in insertion order."""
        return [a for e in self.entities for a in e.actions]
