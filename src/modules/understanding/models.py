from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedAction:
    action_type: str
    actor: str = ""
    target: str = ""
    raw_text: str = ""
    order: int = 0
    duration: float | None = None
    modifier: str = ""


@dataclass(slots=True)
class ParsedEntity:
    name: str
    object_type: str
    is_actor: bool = False
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
    spatial_relations: list[SpatialRelation] = field(default_factory=list)
    duration: float = 5.0
    duration_explicit: bool = False  # True when prompt contained an explicit duration

    @property
    def actions(self) -> list[ParsedAction]:
        """Flat view of all entity actions in insertion order."""
        return [a for e in self.entities for a in e.actions]
