from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlannerConfig:
    random_layout: bool = False
    base_duration: float = 5.0  # fallback when prompt has no explicit durations
    duration_jitter: float = 0.0
    actor_dist: float = 1.5
    obj_space: float = 0.4
    fall_height: float = 1.5
    ground_height: float = 0.5
    default_size: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    default_mass: float = 1.0
    close_action_dist: dict[str, float] = field(
        default_factory=lambda: {"kick": 0.8, "pick_up": 0.5}
    )
    random_range: float = 2.0
    # Random-layout seed. None = nondeterministic (different on every run).
    # Set to an int for reproducible ablations.
    random_seed: int | None = None
