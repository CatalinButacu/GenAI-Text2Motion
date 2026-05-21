from __future__ import annotations

import logging
import random

from src.shared.vocabulary import OBJECTS

from .config import PlannerConfig
from .constraint_layout import solveLayout
from .models import PlannedEntity, PlannedScene, Position3D  # noqa: F401 --re-export

log = logging.getLogger(__name__)


def resolvePos(raw, entity, config: PlannerConfig) -> Position3D:
    """Resolve a raw position value to a Position3D, applying per-entity defaults."""
    if isinstance(raw, Position3D):
        return raw

    if isinstance(raw, tuple):
        return Position3D(*raw)

    y = -config.actorDist if getattr(entity, "isActor", False) else 0.0

    return Position3D(0.0, y, config.groundHeight)


def placeObjects(objects: list, actions: list, config: PlannerConfig) -> dict[str, Position3D]:
    pos: dict[str, Position3D] = {}

    for i, obj in enumerate(objects):
        z = objectZ(obj.name, actions, config)
        pos[obj.name] = Position3D(i * config.objSpace, 0.0, z)
    twoObjFall(objects, actions, pos, config)

    return pos


def objectZ(name: str, actions: list, config: PlannerConfig) -> float:
    for a in actions:
        if a.actionType == "fall":
            return config.fallHeight if a.actor == name else config.groundHeight

    return config.groundHeight


def twoObjFall(
    objects: list, actions: list, pos: dict[str, Position3D], config: PlannerConfig
) -> None:
    if len(objects) != 2:
        return

    for a in actions:
        if a.actionType != "fall" or not a.target:
            continue

        falling = next((o.name for o in objects if o.name != a.target), None)

        if falling:
            pos[a.target] = Position3D(0.0, 0.0, config.groundHeight)
            pos[falling] = Position3D(0.0, 0.0, config.fallHeight)
        break


def placeActors(
    actors: list, actions: list, objPos: dict[str, Position3D], config: PlannerConfig
) -> dict[str, Position3D]:
    pos: dict[str, Position3D] = {}

    for actor in actors:
        target, act_type = None, None

        for a in actions:
            if a.actor == actor.name and a.target:
                target, act_type = a.target, a.actionType
                break

        if target and target in objPos:
            tp = objPos[target]
            d = config.closeActionDist.get(act_type or "", config.actorDist)
            pos[actor.name] = Position3D(tp.x, tp.y - d, 0.0)
        else:
            pos[actor.name] = Position3D(0.0, -config.actorDist, 0.0)

    return pos


class ScenePlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan(self, parsedScene) -> PlannedScene:
        return self.planParsed(parsedScene)

    def computeDuration(self, raw: float, explicit: bool) -> float:
        if self.config.durationJitter > 0 and not explicit:
            return max(
                1.0, raw + random.uniform(-self.config.durationJitter, self.config.durationJitter)
            )

        return raw

    def planParsed(self, parsedScene) -> PlannedScene:
        rawDuration = getattr(parsedScene, "duration", self.config.baseDuration)
        durationExplicit = getattr(parsedScene, "durationExplicit", False)
        duration = self.computeDuration(rawDuration, durationExplicit)

        if self.config.randomLayout:
            r, rng = self.config.randomRange, random.Random(self.config.randomSeed)
            posMap = {
                e.name: Position3D(
                    rng.uniform(-r, r), rng.uniform(-r, r), self.config.groundHeight
                )
                for e in parsedScene.entities
            }

            return self.buildPlanned(
                parsedScene.entities,
                posMap,
                "random",
                duration,
                actions=parsedScene.actions,
            )

        relations = getattr(parsedScene, "spatialRelations", [])
        hasTriples = bool(relations) and getattr(relations[0], "subject", None) is not None

        if hasTriples:
            if solved := trySolve([e.name for e in parsedScene.entities], relations):
                return self.buildPlanned(
                    parsedScene.entities,
                    solved,
                    "constraint",
                    duration,
                    actions=parsedScene.actions,
                )

        actors = [e for e in parsedScene.entities if e.isActor]
        objects = [e for e in parsedScene.entities if not e.isActor]
        objPos = placeObjects(objects, parsedScene.actions, self.config)
        actPos = placeActors(actors, parsedScene.actions, objPos, self.config)

        return self.buildPlanned(
            parsedScene.entities,
            {**objPos, **actPos},
            "row",
            duration,
            actions=parsedScene.actions,
        )

    def buildPlanned(
        self,
        entities,
        posMap: dict,
        mode: str,
        duration: float = 5.0,
        actions: list | None = None,
    ) -> PlannedScene:
        planned: list[PlannedEntity] = []

        for e in entities:
            raw = posMap.get(e.name) if hasattr(e, "name") else None
            pos = resolvePos(raw, e, self.config)
            od = OBJECTS.get(getattr(e, "objectType", "object"))
            isActor = getattr(e, "isActor", False)
            planned.append(
                PlannedEntity(
                    name=e.name,
                    objectType=getattr(e, "objectType", "object"),
                    position=pos,
                    skin=getattr(e, "skin", None),
                    isActor=isActor,
                    size=od.defaultSize if od else self.config.defaultSize,
                    mass=od.defaultMass if od else self.config.defaultMass,
                )
            )
        log.info("ScenePlanner (%s): %d entities", mode, len(planned))

        return PlannedScene(entities=planned, duration=duration, actions=actions or [])


def trySolve(entityNames: list[str], relations: list) -> dict[str, tuple]:
    """Run the L-BFGS-B layout solver. Returns an empty dict when there is nothing
    to solve (no entities or no relations); raises if the solver itself fails.
    """
    if not entityNames or not relations:
        return {}

    return solveLayout(entityNames, relations)
