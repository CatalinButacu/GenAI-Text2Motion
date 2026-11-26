from __future__ import annotations

import csv
import logging
from pathlib import Path

from text2motion.motion.contracts import (
    TextAnnotation,
)

logger = logging.getLogger(__name__)


RARE_VERB_MIN_CLIPS = 5


RARE_VERB_POWER = 0.5


MOTION_CLASS_COLUMN = "class_10"


MOTION_CLASS_POWER = 0.5


NO_MOTION_CLASS = -1


VERB_BALANCE_MIN_CLIPS = 20


NO_VERB_GROUP = ""


NOT_ACTION_VERBS = frozenset(
    {"be", "have", "do", "seem", "appear", "begin", "start", "continue", "try", "use"}
)


TAGGER_ERROR_VERBS = frozenset({"left", "side", "back", "clockwise", "swinge", "outstretche"})


def read_motion_classes(path: Path) -> dict[str, int]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        classes = {row["verb"]: int(row[MOTION_CLASS_COLUMN]) for row in csv.DictReader(handle)}
    if not classes:
        raise ValueError(f"the motion class map {path} has no verb")
    return classes


def caption_verbs(annotation: TextAnnotation) -> frozenset[str]:
    verbs = set()
    for token in annotation.tokens:
        word, _, tag = token.rpartition("/")
        if tag == "VERB" and word:
            verbs.add(word.lower())
    return frozenset(verbs) - NOT_ACTION_VERBS - TAGGER_ERROR_VERBS
