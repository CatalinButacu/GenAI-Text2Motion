from __future__ import annotations

import re as re

from src.shared.constants import SCENE_COLORS

SEQUENCE_MARKERS: list[tuple[str, bool]] = [
    ("and then", False),
    ("after that", False),
    ("after which", False),
    ("and after", False),
    ("and finally", False),
    ("subsequently", False),
    ("followed by", False),
    ("then", False),
    ("next", False),
    ("finally", False),
    ("before", False),
    ("simultaneously", True),
    ("at the same time", True),
    ("while", True),
    ("as well as", True),
]

DURATION_RE = re.compile(
    # matches: "for 3s", "for 2.5 seconds", "for 90 sec",
    #          "for 2 minutes", "for 1.5 min",
    #          "3 seconds" / "2 minutes" (without leading 'for')
    r"(?:for\s+)?(\d+(?:\.\d+)?)\s*" r"(?:s(?:ec(?:ond)?s?)?|min(?:ute)?s?)\b",
    re.IGNORECASE,
)

MODIFIER_WORDS: frozenset[str] = frozenset(
    {
        "quickly",
        "slowly",
        "fast",
        "aggressively",
        "nervously",
        "happily",
        "tiredly",
        "carefully",
        "gently",
        "violently",
        "angrily",
        "stressed",
        "anxiously",
        "calmly",
        "frantically",
        "lazily",
        "gracefully",
        "awkwardly",
        "smoothly",
        "hesitantly",
        "confidently",
        "fearfully",
    }
)

SPATIAL_RELATIONS: dict[str, str] = {
    "on top of": "ON",
    "on": "ON",
    "above": "ABOVE",
    "over": "ABOVE",
    "beneath": "UNDER",
    "under": "UNDER",
    "below": "UNDER",
    "in front of": "IN_FRONT_OF",
    "behind": "BEHIND",
    "beside": "BESIDE",
    "next to": "BESIDE",
    "near": "NEAR",
    "left of": "LEFT_OF",
    "right of": "RIGHT_OF",
    "inside": "INSIDE",
    "in": "INSIDE",
}

RGBA_TO_NAME: dict[tuple, str] = {v: k for k, v in SCENE_COLORS.items()}

QUANTIFIERS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "both": 2,
    "couple": 2,
    "few": 3,
    "several": 4,
    "many": 5,
}

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
