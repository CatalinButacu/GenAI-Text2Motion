from __future__ import annotations

import logging

import spacy

from src.shared.vocabulary import register_nlp

from .actions import (
    backfill_actor_targets,
    compute_scene_duration,
    extract_actions,
    extract_duration,
    extract_modifier,
    propagate_rename,
)
from .config import ParserConfig
from .entities import (
    build_entity_ruler,
    extract_entities,
    extract_spatial,
    has_anaphoric_humanoid_marker,
    make_anaphoric_humanoid,
    register_entity,
)
from .models import ParsedAction, ParsedEntity, ParsedScene
from .parsing_utils import split_into_clauses

log = logging.getLogger(__name__)


class SpacyParser:
    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        self.nlp = spacy.load(self.config.spacy.model)
        build_entity_ruler(self.nlp)
        register_nlp(self.nlp)
        log.info("[M1] SpacyParser loaded (%s)", self.nlp.meta["name"])

    def entities_for_clause(self, doc, _clause, registry, actions):
        found: list[ParsedEntity] = []

        for e in extract_entities(doc):
            final_name, renamed_old = register_entity(e, registry)
            found.append(registry[final_name])

            if renamed_old:
                propagate_rename(actions, renamed_old, f"{renamed_old}_1")

        if not found and has_anaphoric_humanoid_marker(doc):
            if any(e.object_type == "humanoid" for e in registry.values()):
                anon = make_anaphoric_humanoid()
                final_name, renamed_old = register_entity(anon, registry)

                if renamed_old:
                    propagate_rename(actions, renamed_old, f"{renamed_old}_1")
                found.append(registry[final_name])

        return found

    def actions_for_clause(self, doc, entities, order, clause):
        return extract_actions(
            doc,
            entities,
            order,
            extract_duration(clause),
            extract_modifier(clause),
            clause,
        )

    def build_scene(self, prompt, registry, actions):
        entities = list(registry.values())
        spatial = extract_spatial(prompt.lower(), entities)
        backfill_actor_targets(actions, entities)
        actions.sort(key=lambda a: a.order)
        entity_map = {e.name: e for e in entities}
        seen: dict[str, set] = {e.name: set() for e in entities}

        for a in actions:
            if a.actor not in entity_map:
                continue

            key = (a.action_type, a.target, a.order)

            if key not in seen[a.actor]:
                seen[a.actor].add(key)
                entity_map[a.actor].actions.append(a)
        duration, duration_explicit = compute_scene_duration(actions)
        log.info(
            "[M1] %d entities, %d actions, %d spatial", len(entities), len(actions), len(spatial)
        )

        return ParsedScene(
            prompt=prompt,
            entities=entities,
            spatial_relations=spatial,
            duration=duration,
            duration_explicit=duration_explicit,
        )

    def parse(self, prompt: str) -> ParsedScene:
        prompt = (prompt or "").strip()

        if not prompt:
            return ParsedScene(prompt=prompt)

        registry: dict[str, ParsedEntity] = {}
        actions: list[ParsedAction] = []
        order = 0

        for clause, is_concurrent in split_into_clauses(prompt.lower()):
            if actions and not is_concurrent:
                order += 1
            doc = self.nlp(clause)
            entities = self.entities_for_clause(doc, clause, registry, actions)
            actions.extend(self.actions_for_clause(doc, entities, order, clause))

        return self.build_scene(prompt, registry, actions)
