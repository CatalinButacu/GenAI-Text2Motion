from __future__ import annotations

import dataclasses
import logging
import re
from collections import defaultdict

from src.shared.constants import SCENE_COLORS
from src.shared.vocabulary import ACTIONS, OBJECTS, ObjectCategory

from .actions import resolveNoun
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


def tokenColor(token) -> tuple | None:
    for child in token.children:
        if child.dep_ == "amod" and child.lemma_.lower() in SCENE_COLORS:
            return SCENE_COLORS[child.lemma_.lower()]

    return None


def tokenCount(token) -> int:
    for child in token.children:
        if child.dep_ == "nummod":
            try:
                return max(1, int(child.text))

            except ValueError:
                return WORD_TO_COUNT.get(child.lemma_.lower(), 1)

    return 1


def buildEntity(objType: str, isActor: bool, color) -> ParsedEntity:
    colorName = RGBA_TO_NAME.get(color, "") if color else ""

    return ParsedEntity(
        name=f"{colorName}_{objType}".strip("_") if colorName else objType,
        objectType=objType,
        isActor=isActor,
        skin=colorName or None,
    )


def addEntity(
    entity: ParsedEntity,
    count: int,
    seenNames: set[str],
    seenTypes: set[str],
    result: list[ParsedEntity],
) -> None:
    if count > 1:
        if f"{entity.name}_1" in seenNames:
            return

        seenTypes.add(entity.objectType)

        for i in range(1, count + 1):
            n = f"{entity.name}_{i}"
            seenNames.add(n)
            result.append(dataclasses.replace(entity, name=n, actions=[]))
    else:
        if entity.name in seenNames:
            return

        seenNames.add(entity.name)
        seenTypes.add(entity.objectType)
        result.append(entity)


def spanHasAnaphoricDet(entRoot) -> bool:
    return any(c.dep_ == "det" and c.lower_ in ANAPHORIC_HUMANOID for c in entRoot.children)


def buildEntityRuler(nlp) -> None:
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


def registerEntity(e: ParsedEntity, registry: dict[str, ParsedEntity]) -> tuple[str, str | None]:
    base = e.name

    if base not in registry:
        if f"{base}_1" in registry:
            i = 2

            while f"{base}_{i}" in registry:
                i += 1
            newName = f"{base}_{i}"
            registry[newName] = dataclasses.replace(e, name=newName, actions=[])

            return newName, None

        registry[base] = e

        return base, None

    existing = registry.pop(base)
    registry[f"{base}_1"] = dataclasses.replace(existing, name=f"{base}_1", actions=[])
    registry[f"{base}_2"] = dataclasses.replace(e, name=f"{base}_2", actions=[])

    return f"{base}_2", base


def isHumanoidObjType(objType: str) -> bool:
    odef = OBJECTS.get(objType)

    return bool(odef and odef.category == ObjectCategory.HUMANOID)


def extractEntities(doc) -> list[ParsedEntity]:
    seenNames: set[str] = set()
    seenTypes: set[str] = set()
    result: list[ParsedEntity] = []

    spanList: list[tuple[ParsedEntity, int, bool]] = []

    for ent in doc.ents:
        if not ent.label_.startswith("VOCAB_OBJ:"):
            continue

        objType = ent.label_.split(":", 1)[1]
        entity = buildEntity(objType, isHumanoidObjType(objType), tokenColor(ent.root))
        spanList.append((entity, tokenCount(ent.root), spanHasAnaphoricDet(ent.root)))

    nameGroups: dict[str, list] = defaultdict(list)

    for entry in spanList:
        nameGroups[entry[0].name].append(entry)

    for base_name, group in nameGroups.items():
        first_entity, first_count, _ = group[0]
        # Use the explicit numeric count from the first mention only.
        # Multiple NLP spans of the same object type in one clause may be
        # coreferential (e.g. "a ball rolls and hits a ball") -- dedup to 1.
        # Genuinely distinct types appear in separate groups.
        addEntity(first_entity, first_count, seenNames, seenTypes, result)

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue

        objType = resolveNoun(token.text, token.lemma_)

        if not objType or objType in seenTypes:
            continue

        entity = buildEntity(objType, isHumanoidObjType(objType), tokenColor(token))
        addEntity(entity, tokenCount(token), seenNames, seenTypes, result)

    return result


def hasAnaphoricHumanoidMarker(doc) -> bool:
    return any(t.lower_ in ANAPHORIC_HUMANOID for t in doc)


def makeAnaphoricHumanoid() -> ParsedEntity:
    return ParsedEntity(name="humanoid", objectType="humanoid", isActor=True)


def extractSpatial(text: str, entities: list[ParsedEntity]) -> list[SpatialRelation]:
    result: list[SpatialRelation] = []
    consumed: list[tuple[int, int]] = []  # (start, end) of matched prep spans

    for prep, relation in sorted(SPATIAL_RELATIONS.items(), key=lambda x: -len(x[0])):
        idx = text.find(prep)

        if idx < 0:
            continue

        end = idx + len(prep)

        if any(s <= idx < e for s, e in consumed):
            continue

        subj = findEntity(text[:idx], entities)
        obj = findEntity(text[end:], entities)

        if subj and obj:
            result.append(
                SpatialRelation(subject=subj, predicate=prep, relation=relation, object=obj)
            )
            consumed.append((idx, end))

    return result


def findEntity(text: str, entities: list[ParsedEntity]) -> str:
    for e in entities:
        odef = OBJECTS.get(e.objectType)

        if odef:
            for kw in odef.keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return e.name

    return ""
