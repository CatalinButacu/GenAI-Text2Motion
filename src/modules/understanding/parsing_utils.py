from __future__ import annotations

import re

from src.shared.constants import (
    ANAPHORIC_HUMANOID,
    CONSTS,
    DURATION_RE,
    MODIFIER_WORDS,
    SEQUENCE_MARKERS,
    SPATIAL_RELATIONS,
)

RGBA_TO_NAME: dict[tuple, str] = {v: k for k, v in CONSTS.scene.colors.items()}

SUBJ_DEPS: frozenset[str] = frozenset({"nsubj", "nsubjpass", "csubj", "expl"})
OBJ_DEPS: frozenset[str] = frozenset({"dobj", "obj", "pobj", "attr", "oprd"})

CLAUSE_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m, _ in SEQUENCE_MARKERS) + r")\b",
    re.IGNORECASE,
)
CONCURRENT_MAP: dict[str, bool] = {m.lower(): c for m, c in SEQUENCE_MARKERS}


def split_into_clauses(text: str) -> list[tuple[str, bool]]:
    clauses: list[tuple[str, bool]] = []
    concurrent = False

    for part in CLAUSE_RE.split(text):
        part = part.strip(" ,;.")

        if not part:
            continue

        if part.lower() in CONCURRENT_MAP:
            concurrent = CONCURRENT_MAP[part.lower()]
        else:
            clauses.append((part, concurrent))
            concurrent = False

    return clauses or [(text, False)]
