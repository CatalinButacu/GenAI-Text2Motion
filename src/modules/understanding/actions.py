from __future__ import annotations

import re
from itertools import groupby

from src.shared.constants import CONSTS
from src.shared.vocab import ACTIONS, OBJECTS, resolve_action

from .models import ParsedAction, ParsedEntity
from .parsing_utils import DURATION_RE, MODIFIER_WORDS, OBJ_DEPS, SUBJ_DEPS


def build_action_lemma_map() -> dict[str, str]:
    result: dict[str, str] = {}

    for action_name, act_def in ACTIONS.items():
        for kw in act_def.keywords:
            result[kw.lower()] = action_name

            for suffix in ("ing", "ed", "es", "s"):
                if kw.endswith(suffix) and len(kw) - len(suffix) >= 3:
                    result[kw[: -len(suffix)]] = action_name

    return result


LEMMA_TO_ACTION: dict[str, str] = build_action_lemma_map()
LEMMA_TO_OBJECT: dict[str, str] = {
    kw.lower(): obj_name for obj_name, obj_def in OBJECTS.items() for kw in obj_def.keywords
}


def resolve_verb(lemma: str, raw: str) -> str | None:
    for candidate in (lemma.lower(), raw.lower()):
        if candidate in LEMMA_TO_ACTION:
            return LEMMA_TO_ACTION[candidate]

    act_def = resolve_action(lemma)

    return act_def.name if act_def else None


def resolve_noun(token_text: str, lemma: str) -> str | None:
    for candidate in (token_text.lower(), lemma.lower()):
        if candidate in LEMMA_TO_OBJECT:
            return LEMMA_TO_OBJECT[candidate]

    return None


def name_for_noun(token, entities) -> str:
    resolved = resolve_noun(token.text, token.lemma_)

    if not resolved:
        return ""

    for e in entities:
        if e.object_type == resolved:
            return e.name

    return ""


def names_for_token(t, entities) -> list[str]:
    resolved = resolve_noun(t.text, t.lemma_)

    if not resolved:
        return []

    return [e.name for e in entities if e.object_type == resolved]


def add_unique(subjects: list[str], names: list[str]) -> None:
    for name in names:
        if name not in subjects:
            subjects.append(name)


def collect_subj_token(tkn, entities, subjects: list[str]) -> None:
    for child in tkn.children:
        if child.dep_ in SUBJ_DEPS:
            add_unique(subjects, names_for_token(child, entities))
            # coordinated subjects: "a person and a robot walk"
            for conj in child.children:
                if conj.dep_ == "conj":
                    add_unique(subjects, names_for_token(conj, entities))


def collect_agent_token(tkn, entities, subjects: list[str]) -> None:
    """Passive voice: 'ball is kicked by person' -- agent prep gives real actor."""
    for child in tkn.children:
        if child.dep_ == "agent":
            for gc in child.children:
                add_unique(subjects, names_for_token(gc, entities))


def collect_compound_token(tkn, entities, subjects: list[str]) -> None:
    for child in tkn.children:
        if child.dep_ == "compound":
            add_unique(subjects, names_for_token(child, entities))


def dep_subjects(token, entities) -> list[str]:
    subjects: list[str] = []
    # Coordinated verbs ("rolls and hits"): grammatical subject sits on the head verb.
    target = token.head if token.dep_ == "conj" else token

    # Prefer agent (passive) over nsubjpass when both present
    collect_agent_token(target, entities, subjects)

    if not subjects:
        collect_subj_token(target, entities, subjects)

    # Fallback: en_core_web_sm mislabels short sentences with subject as compound dep.
    if not subjects:
        collect_compound_token(target, entities, subjects)

    return subjects


def dep_arg_via_prep(child, entities, deps: frozenset[str]) -> str:
    """Search grandchildren of a prep token for an object dep match."""
    for grandchild in child.children:
        if grandchild.dep_ in deps:
            name = name_for_noun(grandchild, entities)

            if name:
                return name

    return ""


def dep_arg(token, entities, deps: frozenset[str]) -> str:
    for child in token.children:
        if child.dep_ in deps:
            name = name_for_noun(child, entities)

            if name:
                return name

        if child.dep_ == "prep":
            name = dep_arg_via_prep(child, entities, deps)

            if name:
                return name

    return ""


def is_negated(token) -> bool:
    return any(c.dep_ == "neg" for c in token.children)


def emit_resolved(at, head, entities, actors, objects, emit_fn) -> None:
    adef = ACTIONS.get(at)
    subjects = dep_subjects(head, entities) or (actors[:1] if actors else [""])
    target = dep_arg(head, entities, OBJ_DEPS) or (
        objects[0] if adef and adef.requires_target and objects else ""
    )

    for actor in subjects:
        emit_fn(at, actor, target)


def emit_span_actions(ent, entities, actors, objects, emit_fn) -> None:
    """Pass 1 helper: emit actions for a single EntityRuler span."""
    if not ent.label_.startswith("VOCAB_ACT:"):
        return

    if is_negated(ent.root):
        return

    at = ent.label_.split(":", 1)[1]
    emit_resolved(at, ent.root, entities, actors, objects, emit_fn)


def emit_token_actions(token, entities, actors, objects, emit_fn) -> None:
    """Pass 2 helper: emit actions for a single verb token."""
    if token.pos_ != "VERB" or is_negated(token):
        return

    at = resolve_verb(token.lemma_, token.text)

    if at is None:
        return

    emit_resolved(at, token, entities, actors, objects, emit_fn)


def extract_actions(doc, entities, order, duration, modifier, clause_text) -> list[ParsedAction]:
    actors = [e.name for e in entities if e.is_actor]
    objects = [e.name for e in entities if not e.is_actor]
    result: list[ParsedAction] = []
    seen: set[tuple[str, str]] = set()

    def emit(at: str, actor: str, target: str) -> None:
        key = (at, actor)

        if key in seen:
            return

        seen.add(key)
        result.append(
            ParsedAction(
                action_type=at,
                actor=actor,
                target=target,
                raw_text=clause_text.strip(),
                order=order,
                duration=duration,
                modifier=modifier,
            )
        )

    for ent in doc.ents:
        emit_span_actions(ent, entities, actors, objects, emit)

    for token in doc:
        emit_token_actions(token, entities, actors, objects, emit)

    return result


def extract_duration(text: str) -> float | None:
    m = DURATION_RE.search(text)

    if not m:
        return None

    val = float(m.group(1))

    if re.search(r"\bmin", m.group(0), re.IGNORECASE):
        val *= 60.0

    return val


def extract_modifier(text: str) -> str:
    # sorted() gives deterministic order; join all matches so none are lost
    words = set(re.findall(r"[a-z]+", text.lower()))
    matches = sorted(w for w in words if w in MODIFIER_WORDS)

    return " ".join(matches)


def backfill_actor_targets(actions: list[ParsedAction], entities: list[ParsedEntity]) -> None:
    actors = [e.name for e in entities if e.is_actor]
    objects = [e.name for e in entities if not e.is_actor]

    for a in actions:
        if not a.actor and actors:
            a.actor = actors[0]
        adef = ACTIONS.get(a.action_type)

        if not a.target and adef and adef.requires_target and objects:
            a.target = objects[0]


def propagate_rename(actions: list[ParsedAction], old: str, new: str) -> None:
    """Update actor/target references after an entity is renamed (e.g. humanoid -> humanoid_1)."""
    for a in actions:
        if a.actor == old:
            a.actor = new

        if a.target == old:
            a.target = new


def compute_scene_duration(actions: list[ParsedAction]) -> tuple[float, bool]:
    """Return (total_duration, duration_explicit) from per-action slot durations."""
    if not any(a.duration is not None for a in actions):
        return CONSTS.runtime.default_clip_duration_s, False

    slots = [
        max(a.duration for a in group if a.duration is not None)
        for slot_key, slot_iter in groupby(actions, key=lambda a: a.order)
        for group in [[*slot_iter]]
        if any(a.duration is not None for a in group)
    ]

    return (sum(slots) if slots else CONSTS.runtime.default_clip_duration_s), True
