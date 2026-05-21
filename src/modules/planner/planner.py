from __future__ import annotations

import logging
import random

from src.shared.vocabulary import OBJECTS

from .config import PlannerConfig
from .constraint_layout import solve_layout
from .models import PlannedEntity, PlannedScene, Position3D  # noqa: F401 --re-export

log = logging.getLogger(__name__)


def resolve_pos(raw, entity, config: PlannerConfig) -> Position3D:
    """Resolve a raw position value to a Position3D, applying per-entity defaults."""
    if isinstance(raw, Position3D):
        return raw

    if isinstance(raw, tuple):
        return Position3D(*raw)

    y = -config.actor_dist if getattr(entity, "is_actor", False) else 0.0

    return Position3D(0.0, y, config.ground_height)


def place_objects(objects: list, actions: list, config: PlannerConfig) -> dict[str, Position3D]:
    pos: dict[str, Position3D] = {}

    for i, obj in enumerate(objects):
        z = object_z(obj.name, actions, config)
        pos[obj.name] = Position3D(i * config.obj_space, 0.0, z)
    two_obj_fall(objects, actions, pos, config)

    return pos


def object_z(name: str, actions: list, config: PlannerConfig) -> float:
    for a in actions:
        if a.action_type == "fall":
            return config.fall_height if a.actor == name else config.ground_height

    return config.ground_height


def two_obj_fall(
    objects: list, actions: list, pos: dict[str, Position3D], config: PlannerConfig
) -> None:
    if len(objects) != 2:
        return

    for a in actions:
        if a.action_type != "fall" or not a.target:
            continue

        falling = next((o.name for o in objects if o.name != a.target), None)

        if falling:
            pos[a.target] = Position3D(0.0, 0.0, config.ground_height)
            pos[falling] = Position3D(0.0, 0.0, config.fall_height)
        break


def place_actors(
    actors: list, actions: list, obj_pos: dict[str, Position3D], config: PlannerConfig
) -> dict[str, Position3D]:
    pos: dict[str, Position3D] = {}

    for actor in actors:
        target, act_type = None, None

        for a in actions:
            if a.actor == actor.name and a.target:
                target, act_type = a.target, a.action_type
                break

        if target and target in obj_pos:
            tp = obj_pos[target]
            d = config.close_action_dist.get(act_type or "", config.actor_dist)
            pos[actor.name] = Position3D(tp.x, tp.y - d, 0.0)
        else:
            pos[actor.name] = Position3D(0.0, -config.actor_dist, 0.0)

    return pos


class ScenePlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan(self, parsed_scene) -> PlannedScene:
        return self.plan_parsed(parsed_scene)

    def compute_duration(self, raw: float, explicit: bool) -> float:
        if self.config.duration_jitter > 0 and not explicit:
            return max(
                1.0, raw + random.uniform(-self.config.duration_jitter, self.config.duration_jitter)
            )

        return raw

    def plan_parsed(self, parsed_scene) -> PlannedScene:
        raw_duration = getattr(parsed_scene, "duration", self.config.base_duration)
        duration_explicit = getattr(parsed_scene, "duration_explicit", False)
        duration = self.compute_duration(raw_duration, duration_explicit)

        if self.config.random_layout:
            r, rng = self.config.random_range, random.Random(self.config.random_seed)
            pos_map = {
                e.name: Position3D(
                    rng.uniform(-r, r), rng.uniform(-r, r), self.config.ground_height
                )
                for e in parsed_scene.entities
            }

            return self.build_planned(
                parsed_scene.entities,
                pos_map,
                "random",
                duration,
                actions=parsed_scene.actions,
            )

        relations = getattr(parsed_scene, "spatial_relations", [])
        has_triples = bool(relations) and getattr(relations[0], "subject", None) is not None

        if has_triples:
            if solved := try_solve([e.name for e in parsed_scene.entities], relations):
                return self.build_planned(
                    parsed_scene.entities,
                    solved,
                    "constraint",
                    duration,
                    actions=parsed_scene.actions,
                )

        actors = [e for e in parsed_scene.entities if e.is_actor]
        objects = [e for e in parsed_scene.entities if not e.is_actor]
        obj_pos = place_objects(objects, parsed_scene.actions, self.config)
        act_pos = place_actors(actors, parsed_scene.actions, obj_pos, self.config)

        return self.build_planned(
            parsed_scene.entities,
            {**obj_pos, **act_pos},
            "row",
            duration,
            actions=parsed_scene.actions,
        )

    def build_planned(
        self,
        entities,
        pos_map: dict,
        mode: str,
        duration: float = 5.0,
        actions: list | None = None,
    ) -> PlannedScene:
        planned: list[PlannedEntity] = []

        for e in entities:
            raw = pos_map.get(e.name) if hasattr(e, "name") else None
            pos = resolve_pos(raw, e, self.config)
            od = OBJECTS.get(getattr(e, "object_type", "object"))
            is_actor = getattr(e, "is_actor", False)
            planned.append(
                PlannedEntity(
                    name=e.name,
                    object_type=getattr(e, "object_type", "object"),
                    position=pos,
                    skin=getattr(e, "skin", None),
                    is_actor=is_actor,
                    size=od.default_size if od else self.config.default_size,
                    mass=od.default_mass if od else self.config.default_mass,
                )
            )
        log.info("ScenePlanner (%s): %d entities", mode, len(planned))

        return PlannedScene(entities=planned, duration=duration, actions=actions or [])


def try_solve(entity_names: list[str], relations: list) -> dict[str, tuple]:
    """Run the L-BFGS-B layout solver. Returns an empty dict when there is nothing
    to solve (no entities or no relations); raises if the solver itself fails.
    """
    if not entity_names or not relations:
        return {}

    return solve_layout(entity_names, relations)
