from __future__ import annotations

import logging

import spacy

from src.shared.vocabulary import registerNlp

from .actions import (
    backfillActorTargets,
    computeSceneDuration,
    extractActions,
    extractDuration,
    extractModifier,
    propagateRename,
)
from .config import ParserConfig
from .entities import (
    buildEntityRuler,
    extractEntities,
    extractSpatial,
    hasAnaphoricHumanoidMarker,
    makeAnaphoricHumanoid,
    registerEntity,
)
from .models import ParsedAction, ParsedEntity, ParsedScene
from .parsing_utils import splitIntoClauses

log = logging.getLogger(__name__)


class SpacyParser:
    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        self.nlp = spacy.load(self.config.spacy.model)
        buildEntityRuler(self.nlp)
        registerNlp(self.nlp)
        log.info("[M1] SpacyParser loaded (%s)", self.nlp.meta["name"])

    def entitiesForClause(self, doc, _clause, registry, actions):
        found: list[ParsedEntity] = []

        for e in extractEntities(doc):
            final_name, renamed_old = registerEntity(e, registry)
            found.append(registry[final_name])

            if renamed_old:
                propagateRename(actions, renamed_old, f"{renamed_old}_1")

        if not found and hasAnaphoricHumanoidMarker(doc):
            if any(e.objectType == "humanoid" for e in registry.values()):
                anon = makeAnaphoricHumanoid()
                final_name, renamed_old = registerEntity(anon, registry)

                if renamed_old:
                    propagateRename(actions, renamed_old, f"{renamed_old}_1")
                found.append(registry[final_name])

        return found

    def actionsForClause(self, doc, entities, order, clause):
        return extractActions(
            doc,
            entities,
            order,
            extractDuration(clause),
            extractModifier(clause),
            clause,
        )

    def buildScene(self, prompt, registry, actions):
        entities = list(registry.values())
        spatial = extractSpatial(prompt.lower(), entities)
        backfillActorTargets(actions, entities)
        actions.sort(key=lambda a: a.order)
        entityMap = {e.name: e for e in entities}
        seen: dict[str, set] = {e.name: set() for e in entities}

        for a in actions:
            if a.actor not in entityMap:
                continue

            key = (a.actionType, a.target, a.order)

            if key not in seen[a.actor]:
                seen[a.actor].add(key)
                entityMap[a.actor].actions.append(a)
        duration, duration_explicit = computeSceneDuration(actions)
        log.info(
            "[M1] %d entities, %d actions, %d spatial", len(entities), len(actions), len(spatial)
        )

        return ParsedScene(
            prompt=prompt,
            entities=entities,
            spatialRelations=spatial,
            duration=duration,
            durationExplicit=duration_explicit,
        )

    def parse(self, prompt: str) -> ParsedScene:
        prompt = (prompt or "").strip()

        if not prompt:
            return ParsedScene(prompt=prompt)

        registry: dict[str, ParsedEntity] = {}
        actions: list[ParsedAction] = []
        order = 0

        for clause, is_concurrent in splitIntoClauses(prompt.lower()):
            if actions and not is_concurrent:
                order += 1
            doc = self.nlp(clause)
            entities = self.entitiesForClause(doc, clause, registry, actions)
            actions.extend(self.actionsForClause(doc, entities, order, clause))

        return self.buildScene(prompt, registry, actions)
