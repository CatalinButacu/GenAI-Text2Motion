from __future__ import annotations

import re
from itertools import groupby

from src.shared.vocabulary import ACTIONS, OBJECTS, resolveAction

from .models import ParsedAction, ParsedEntity
from .parsing_utils import DURATION_RE, MODIFIER_WORDS, OBJ_DEPS, SUBJ_DEPS


def buildActionLemmaMap() -> dict[str, str]:
    result: dict[str, str] = {}

    for action_name, act_def in ACTIONS.items():
        for kw in act_def.keywords:
            result[kw.lower()] = action_name

            for suffix in ("ing", "ed", "es", "s"):
                if kw.endswith(suffix) and len(kw) - len(suffix) >= 3:
                    result[kw[: -len(suffix)]] = action_name

    return result


LEMMA_TO_ACTION: dict[str, str] = buildActionLemmaMap()
LEMMA_TO_OBJECT: dict[str, str] = {
    kw.lower(): obj_name for obj_name, obj_def in OBJECTS.items() for kw in obj_def.keywords
}


def resolveVerb(lemma: str, raw: str) -> str | None:
    for candidate in (lemma.lower(), raw.lower()):
        if candidate in LEMMA_TO_ACTION:
            return LEMMA_TO_ACTION[candidate]

    actDef = resolveAction(lemma)

    return actDef.name if actDef else None


def resolveNoun(tokenText: str, lemma: str) -> str | None:
    for candidate in (tokenText.lower(), lemma.lower()):
        if candidate in LEMMA_TO_OBJECT:
            return LEMMA_TO_OBJECT[candidate]

    return None


def nameForNoun(token, entities) -> str:
    resolved = resolveNoun(token.text, token.lemma_)

    if not resolved:
        return ""

    for e in entities:
        if e.objectType == resolved:
            return e.name

    return ""


def namesForToken(t, entities) -> list[str]:
    resolved = resolveNoun(t.text, t.lemma_)

    if not resolved:
        return []

    return [e.name for e in entities if e.objectType == resolved]


def addUnique(subjects: list[str], names: list[str]) -> None:
    for name in names:
        if name not in subjects:
            subjects.append(name)


def collectSubjToken(tkn, entities, subjects: list[str]) -> None:
    for child in tkn.children:
        if child.dep_ in SUBJ_DEPS:
            addUnique(subjects, namesForToken(child, entities))
            # coordinated subjects: "a person and a robot walk"
            for conj in child.children:
                if conj.dep_ == "conj":
                    addUnique(subjects, namesForToken(conj, entities))


def collectAgentToken(tkn, entities, subjects: list[str]) -> None:
    """Passive voice: 'ball is kicked by person' -- agent prep gives real actor."""
    for child in tkn.children:
        if child.dep_ == "agent":
            for gc in child.children:
                addUnique(subjects, namesForToken(gc, entities))


def collectCompoundToken(tkn, entities, subjects: list[str]) -> None:
    for child in tkn.children:
        if child.dep_ == "compound":
            addUnique(subjects, namesForToken(child, entities))


def depSubjects(token, entities) -> list[str]:
    subjects: list[str] = []
    # For coordinated verbs (e.g. "rolls and hits"), the grammatical subject
    # is attached to the head verb, not the conj verb.
    target = token.head if token.dep_ == "conj" else token

    # Prefer agent (passive) over nsubjpass when both present
    collectAgentToken(target, entities, subjects)

    if not subjects:
        collectSubjToken(target, entities, subjects)

    # Fallback: spaCy's en_core_web_sm mislabels short physics-context sentences
    # (e.g. "a cube slides") as NOUN ROOT with the subject as a "compound" dep.
    if not subjects:
        collectCompoundToken(target, entities, subjects)

    return subjects


def depArgViaPrep(child, entities, deps: frozenset[str]) -> str:
    """Search grandchildren of a prep token for an object dep match."""
    for grandchild in child.children:
        if grandchild.dep_ in deps:
            name = nameForNoun(grandchild, entities)

            if name:
                return name

    return ""


def depArg(token, entities, deps: frozenset[str]) -> str:
    for child in token.children:
        if child.dep_ in deps:
            name = nameForNoun(child, entities)

            if name:
                return name

        if child.dep_ == "prep":
            name = depArgViaPrep(child, entities, deps)

            if name:
                return name

    return ""


def isNegated(token) -> bool:
    return any(c.dep_ == "neg" for c in token.children)


def emitSpanActions(ent, entities, actors, objects, emitFn) -> None:
    """Pass 1 helper: emit actions for a single EntityRuler span."""
    if not ent.label_.startswith("VOCAB_ACT:"):
        return

    at = ent.label_.split(":", 1)[1]

    if isNegated(ent.root):
        return

    adef = ACTIONS.get(at)
    head = ent.root
    subjects = depSubjects(head, entities) or (actors[:1] if actors else [""])
    target = depArg(head, entities, OBJ_DEPS) or (
        objects[0] if adef and adef.requiresTarget and objects else ""
    )

    for actor in subjects:
        emitFn(at, actor, target)


def emitTokenActions(token, entities, actors, objects, emitFn) -> None:
    """Pass 2 helper: emit actions for a single verb token."""
    if token.pos_ != "VERB" or isNegated(token):
        return

    at = resolveVerb(token.lemma_, token.text)

    if at is None:
        return

    adef = ACTIONS.get(at)
    subjects = depSubjects(token, entities) or (actors[:1] if actors else [""])
    target = depArg(token, entities, OBJ_DEPS) or (
        objects[0] if adef and adef.requiresTarget and objects else ""
    )

    for actor in subjects:
        emitFn(at, actor, target)


def extractActions(doc, entities, order, duration, modifier, clauseText) -> list[ParsedAction]:
    actors = [e.name for e in entities if e.isActor]
    objects = [e.name for e in entities if not e.isActor]
    result: list[ParsedAction] = []
    seen: set[tuple[str, str]] = set()

    def emit(at: str, actor: str, target: str) -> None:
        key = (at, actor)

        if key in seen:
            return

        seen.add(key)
        result.append(
            ParsedAction(
                actionType=at,
                actor=actor,
                target=target,
                rawText=clauseText.strip(),
                order=order,
                duration=duration,
                modifier=modifier,
            )
        )

    for ent in doc.ents:
        emitSpanActions(ent, entities, actors, objects, emit)

    for token in doc:
        emitTokenActions(token, entities, actors, objects, emit)

    return result


def extractDuration(text: str) -> float | None:
    m = DURATION_RE.search(text)

    if not m:
        return None

    val = float(m.group(1))
    # scale minutes to seconds
    if re.search(r"\bmin", m.group(0), re.IGNORECASE):
        val *= 60.0

    return val


def extractModifier(text: str) -> str:
    # sorted() gives deterministic order; join all matches so none are lost
    words = set(re.findall(r"[a-z]+", text.lower()))
    matches = sorted(w for w in words if w in MODIFIER_WORDS)

    return " ".join(matches)


def backfillActorTargets(actions: list[ParsedAction], entities: list[ParsedEntity]) -> None:
    actors = [e.name for e in entities if e.isActor]
    objects = [e.name for e in entities if not e.isActor]

    for a in actions:
        if not a.actor and actors:
            a.actor = actors[0]
        adef = ACTIONS.get(a.actionType)

        if not a.target and adef and adef.requiresTarget and objects:
            a.target = objects[0]


def propagateRename(actions: list[ParsedAction], old: str, new: str) -> None:
    """Update actor/target references after an entity is renamed (e.g. humanoid -> humanoid_1)."""
    for a in actions:
        if a.actor == old:
            a.actor = new

        if a.target == old:
            a.target = new


def computeSceneDuration(actions: list[ParsedAction]) -> tuple[float, bool]:
    """Return (total_duration, duration_explicit) from per-action slot durations."""
    if not any(a.duration is not None for a in actions):
        return 5.0, False

    slots = [
        max(a.duration for a in group if a.duration is not None)
        for _, slot_iter in groupby(actions, key=lambda a: a.order)
        for group in [[*slot_iter]]
        if any(a.duration is not None for a in group)
    ]

    return (sum(slots) if slots else 5.0), True
