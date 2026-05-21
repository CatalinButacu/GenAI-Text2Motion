from __future__ import annotations

import dataclasses
import logging
import re
from collections import defaultdict

from src.shared.constants import SCENE_COLORS
from src.shared.vocabulary import ACTIONS, OBJECTS, ObjectCategory

from .actions import resolve_noun
from .models import ParsedEntity, SpatialRelation
from .parsing_utils import RGBA_TO_NAME, SPATIAL_RELATIONS

log = logging.getLogger(__name__)
RULER_NAME = "vocab_entity_ruler"

WORD_TO_COUNT: dict[str, int] = {
    "two": 2,
    "couple": 2,
    "pair": 2,
    "both": 2,
    "three": 3,
    "triple": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "several": 3,
    "few": 3,
    "multiple": 3,
    "many": 4,
}

ANAPHORIC_HUMANOID: frozenset[str] = frozenset(
    {
        "another",
        "other",
        "someone",
        "somebody",
        "they",
        "both",
        "second",
    }
)


def token_color(token) -> tuple | None:
    for child in token.children:
        if child.dep_ == "amod" and child.lemma_.lower() in SCENE_COLORS:
            return SCENE_COLORS[child.lemma_.lower()]

    return None


def token_count(token) -> int:
    for child in token.children:
        if child.dep_ == "nummod":
            try:
                return max(1, int(child.text))

            except ValueError:
                return WORD_TO_COUNT.get(child.lemma_.lower(), 1)

    return 1


def build_entity(obj_type: str, is_actor: bool, color) -> ParsedEntity:
    color_name = RGBA_TO_NAME.get(color, "") if color else ""

    return ParsedEntity(
        name=f"{color_name}_{obj_type}".strip("_") if color_name else obj_type,
        object_type=obj_type,
        is_actor=is_actor,
        skin=color_name or None,
    )


def add_entity(
    entity: ParsedEntity,
    count: int,
    seen_names: set[str],
    seen_types: set[str],
    result: list[ParsedEntity],
) -> None:
    if count > 1:
        if f"{entity.name}_1" in seen_names:
            return

        seen_types.add(entity.object_type)

        for i in range(1, count + 1):
            n = f"{entity.name}_{i}"
            seen_names.add(n)
            result.append(dataclasses.replace(entity, name=n, actions=[]))
    else:
        if entity.name in seen_names:
            return

        seen_names.add(entity.name)
        seen_types.add(entity.object_type)
        result.append(entity)


def span_has_anaphoric_det(ent_root) -> bool:
    return any(c.dep_ == "det" and c.lower_ in ANAPHORIC_HUMANOID for c in ent_root.children)


def build_entity_ruler(nlp) -> None:
    if nlp.has_pipe(RULER_NAME):
        return

    ruler = nlp.add_pipe(
        "entity_ruler", name=RULER_NAME, before="ner", config={"overwrite_ents": True}
    )
    patterns: list[dict] = []

    for obj_name, obj_def in OBJECTS.items():
        label = f"VOCAB_OBJ:{obj_name}"

        for kw in obj_def.keywords:
            patterns.append({"label": label, "pattern": kw if " " in kw else [{"LEMMA": kw}]})

    for act_name, act_def in ACTIONS.items():
        label = f"VOCAB_ACT:{act_name}"

        for kw in act_def.keywords:
            patterns.append({"label": label, "pattern": kw if " " in kw else [{"LEMMA": kw}]})
    ruler.add_patterns(patterns)
    log.debug("[M1] EntityRuler: %d patterns added", len(patterns))


def register_entity(e: ParsedEntity, registry: dict[str, ParsedEntity]) -> tuple[str, str | None]:
    base = e.name

    if base not in registry:
        if f"{base}_1" in registry:
            i = 2

            while f"{base}_{i}" in registry:
                i += 1
            new_name = f"{base}_{i}"
            registry[new_name] = dataclasses.replace(e, name=new_name, actions=[])

            return new_name, None

        registry[base] = e

        return base, None

    existing = registry.pop(base)
    registry[f"{base}_1"] = dataclasses.replace(existing, name=f"{base}_1", actions=[])
    registry[f"{base}_2"] = dataclasses.replace(e, name=f"{base}_2", actions=[])

    return f"{base}_2", base


def is_humanoid_obj_type(obj_type: str) -> bool:
    odef = OBJECTS.get(obj_type)

    return bool(odef and odef.category == ObjectCategory.HUMANOID)


def extract_entities(doc) -> list[ParsedEntity]:
    seen_names: set[str] = set()
    seen_types: set[str] = set()
    result: list[ParsedEntity] = []

    span_list: list[tuple[ParsedEntity, int, bool]] = []

    for ent in doc.ents:
        if not ent.label_.startswith("VOCAB_OBJ:"):
            continue

        obj_type = ent.label_.split(":", 1)[1]
        entity = build_entity(obj_type, is_humanoid_obj_type(obj_type), token_color(ent.root))
        span_list.append((entity, token_count(ent.root), span_has_anaphoric_det(ent.root)))

    name_groups: dict[str, list] = defaultdict(list)

    for entry in span_list:
        name_groups[entry[0].name].append(entry)

    for base_name, group in name_groups.items():
        first_entity, first_count, _ = group[0]
        # Use the explicit numeric count from the first mention only.
        # Multiple NLP spans of the same object type in one clause may be
        # coreferential (e.g. "a ball rolls and hits a ball") -- dedup to 1.
        # Genuinely distinct types appear in separate groups.
        add_entity(first_entity, first_count, seen_names, seen_types, result)

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue

        obj_type = resolve_noun(token.text, token.lemma_)

        if not obj_type or obj_type in seen_types:
            continue

        entity = build_entity(obj_type, is_humanoid_obj_type(obj_type), token_color(token))
        add_entity(entity, token_count(token), seen_names, seen_types, result)

    return result


def has_anaphoric_humanoid_marker(doc) -> bool:
    return any(t.lower_ in ANAPHORIC_HUMANOID for t in doc)


def make_anaphoric_humanoid() -> ParsedEntity:
    return ParsedEntity(name="humanoid", object_type="humanoid", is_actor=True)


def extract_spatial(text: str, entities: list[ParsedEntity]) -> list[SpatialRelation]:
    result: list[SpatialRelation] = []
    consumed: list[tuple[int, int]] = []  # (start, end) of matched prep spans

    for prep, relation in sorted(SPATIAL_RELATIONS.items(), key=lambda x: -len(x[0])):
        idx = text.find(prep)

        if idx < 0:
            continue

        end = idx + len(prep)

        if any(s <= idx < e for s, e in consumed):
            continue

        subj = find_entity(text[:idx], entities)
        obj = find_entity(text[end:], entities)

        if subj and obj:
            result.append(
                SpatialRelation(subject=subj, predicate=prep, relation=relation, object=obj)
            )
            consumed.append((idx, end))

    return result


def find_entity(text: str, entities: list[ParsedEntity]) -> str:
    for e in entities:
        odef = OBJECTS.get(e.object_type)

        if odef:
            for kw in odef.keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return e.name

    return ""
