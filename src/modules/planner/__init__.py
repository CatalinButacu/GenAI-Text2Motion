from __future__ import annotations

import logging

from src.utils.mem_profile import tracemallocSnapshot

from .config import PlannerConfig
from .models import PlannedEntity, PlannedScene, Position3D
from .planner import ScenePlanner

log = logging.getLogger(__name__)

PLANNER: ScenePlanner | None = None


def invoke(parsed, config: PlannerConfig | None = None) -> PlannedScene:
    """M2: plan a 3D scene from a ParsedScene. Lazily constructs and caches the planner."""
    global PLANNER

    if PLANNER is None:
        PLANNER = ScenePlanner(config)

    with tracemallocSnapshot("M2 plan"):
        planned = PLANNER.plan(parsed)

    log.info("[M2] %d entities positioned", len(planned.entities))

    return planned


__all__ = ["invoke", "PlannerConfig", "PlannedScene", "PlannedEntity", "Position3D"]
