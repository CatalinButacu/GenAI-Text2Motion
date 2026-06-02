from __future__ import annotations

import logging
from functools import cache
from typing import Any

import spacy

from .actions import ACTIONS, ActionDefinition
from .objects import OBJECTS, ObjectDefinition
from .semantic import SemanticActionResolver

log = logging.getLogger(__name__)

# None = not yet loaded, False = tried and unavailable, Language = ready
LEMMA_NLP: Any = None


def register_nlp(nlp) -> None:
    global LEMMA_NLP

    if LEMMA_NLP is None:
        LEMMA_NLP = nlp
        log.debug("spaCy lemmatizer: reusing shared Language instance")


def get_lemma_nlp():
    global LEMMA_NLP

    if LEMMA_NLP is None:
        try:
            LEMMA_NLP = spacy.load("en_core_web_sm", exclude=["parser", "ner", "senter"])
            log.debug("spaCy lemmatizer loaded")
        except Exception as exc:
            log.debug("spaCy lemmatizer unavailable: %s", exc)
            LEMMA_NLP = False

    return None if LEMMA_NLP is False else LEMMA_NLP


def spacy_lemma(word: str) -> str | None:
    nlp = get_lemma_nlp()

    if not nlp:
        return None

    try:
        doc = nlp(word.lower())
        lemma = doc[0].lemma_ if doc else None

        return lemma if lemma and lemma != word.lower() else None

    except Exception:
        return None


def lookup_keyword(keyword: str, registry: dict):
    kw = keyword.lower().strip()
    exact = next((v for v in registry.values() if kw in v.keywords), None)

    if exact:
        return exact

    lemma = spacy_lemma(kw)

    if lemma:
        return next((v for v in registry.values() if lemma in v.keywords), None)

    return None


@cache
def get_action_by_keyword(keyword: str) -> ActionDefinition | None:
    return lookup_keyword(keyword, ACTIONS)


@cache
def get_object_by_keyword(keyword: str) -> ObjectDefinition | None:
    return lookup_keyword(keyword, OBJECTS)


def resolve_action(text: str) -> ActionDefinition | None:
    for word in text.lower().split():
        result = get_action_by_keyword(word)

        if result is not None:
            return result

    result = get_action_by_keyword(text.lower().strip())

    if result is not None:
        return result

    return SemanticActionResolver.get_instance().resolve(text)
