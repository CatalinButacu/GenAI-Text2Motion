from __future__ import annotations

import logging
from functools import cache
from typing import Any

import spacy

from .vocab_actions import ACTIONS, ActionDefinition
from .vocab_objects import OBJECTS, ObjectDefinition
from .vocab_semantic import SemanticActionResolver

log = logging.getLogger(__name__)

# None = not yet loaded, False = tried and unavailable, Language = ready
LEMMA_NLP: Any = None


def registerNlp(nlp) -> None:
    global LEMMA_NLP

    if LEMMA_NLP is None:
        LEMMA_NLP = nlp
        log.debug("[vocab] spaCy lemmatizer: reusing shared Language instance")


def getLemmaNlp():
    global LEMMA_NLP

    if LEMMA_NLP is None:
        try:
            LEMMA_NLP = spacy.load("en_core_web_sm", exclude=["parser", "ner", "senter"])
            log.debug("[vocab] spaCy lemmatizer loaded")
        except Exception as exc:
            log.debug("[vocab] spaCy lemmatizer unavailable: %s", exc)
            LEMMA_NLP = False

    return None if LEMMA_NLP is False else LEMMA_NLP


def spacyLemma(word: str) -> str | None:
    nlp = getLemmaNlp()

    if not nlp:
        return None

    try:
        doc = nlp(word.lower())
        lemma = doc[0].lemma_ if doc else None

        return lemma if lemma and lemma != word.lower() else None

    except Exception:
        return None


@cache
def getActionByKeyword(keyword: str) -> ActionDefinition | None:
    kw = keyword.lower().strip()
    exact = next((a for a in ACTIONS.values() if kw in a.keywords), None)

    if exact:
        return exact

    lemma = spacyLemma(kw)

    if lemma:
        return next((a for a in ACTIONS.values() if lemma in a.keywords), None)

    return None


@cache
def getObjectByKeyword(keyword: str) -> ObjectDefinition | None:
    kw = keyword.lower().strip()
    exact = next((o for o in OBJECTS.values() if kw in o.keywords), None)

    if exact:
        return exact

    lemma = spacyLemma(kw)

    if lemma:
        return next((o for o in OBJECTS.values() if lemma in o.keywords), None)

    return None


def resolveAction(text: str) -> ActionDefinition | None:
    for word in text.lower().split():
        result = getActionByKeyword(word)

        if result is not None:
            return result

    result = getActionByKeyword(text.lower().strip())

    if result is not None:
        return result

    return SemanticActionResolver.getInstance().resolve(text)
