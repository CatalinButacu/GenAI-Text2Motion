from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlannerConfig:
    randomLayout: bool = False
    baseDuration: float = 5.0  # fallback when prompt has no explicit durations
    durationJitter: float = 0.0
    actorDist: float = 1.5
    objSpace: float = 0.4
    fallHeight: float = 1.5
    groundHeight: float = 0.5
    defaultSize: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    defaultMass: float = 1.0
    closeActionDist: dict[str, float] = field(
        default_factory=lambda: {"kick": 0.8, "pick_up": 0.5}
    )
    randomRange: float = 2.0
    # Random-layout seed. None = nondeterministic (different on every run).
    # Set to an int for reproducible ablations.
    randomSeed: int | None = None
