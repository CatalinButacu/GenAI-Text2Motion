#!/usr/bin/env python
"""Hybrid action label matcher.

Three-tier matching:
  Tier 1 – exact string equality (case-insensitive).
  Tier 2 – alias table lookup from data/eval/action_synonyms.json (bidirectional).
  Tier 3 – spaCy lemma equality (semantic mode only).

Two modes
---------
strict   – tier 1 + substring (backward-compat) + alias table.
           Used for hard gate decisions.
semantic – strict + spaCy lemma equality.
           Used for diagnostic reporting.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

SYNONYM_FILE = Path("data/eval/action_synonyms.json")

try:
    import spacy as spacyLib

    spacy_nlp = spacyLib.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    spacy_nlp = None
    SPACY_AVAILABLE = False


@lru_cache(maxsize=1)
def load_alias_table() -> dict[str, frozenset]:
    """Load synonym file and build bidirectional lookup table.

    Each surface form maps to the full synonym group it belongs to,
    making look-up O(1) per token.
    """
    if not SYNONYM_FILE.exists():
        log.warning("Action synonym file not found: %s — alias matching disabled", SYNONYM_FILE)
        return {}

    raw: dict[str, list[str]] = json.loads(SYNONYM_FILE.read_text(encoding="utf-8"))
    table: dict[str, frozenset] = {}

    for canonical, variants in raw.items():
        group = frozenset({canonical.lower()} | {v.lower() for v in variants})

        for member in group:
            table[member] = group

    return table


def canonical_group(token: str, table: dict[str, frozenset]) -> frozenset:
    return table.get(token.lower(), frozenset({token.lower()}))


def lemmatize(text: str) -> str:
    if not SPACY_AVAILABLE or spacy_nlp is None:
        return text.lower()

    return " ".join(t.lemma_ for t in spacy_nlp(text.lower()))


def match_action(expected: str, detected: str, mode: str = "strict") -> bool:
    """Return True if detected satisfies expected under the given mode.

    Parameters
    ----------
    expected, detected:
        Action label strings (lowercased internally).
    mode:
        'strict'   – exact string + substring (backward-compat) + alias table.
        'semantic' – strict + spaCy lemma equality.
    """
    exp = expected.lower().strip()
    det = detected.lower().strip()

    # tier 1: exact
    if exp == det:
        return True

    # backward compat: substring containment
    if exp in det or det in exp:
        return True

    # tier 2: alias table
    table = load_alias_table()
    exp_group = canonical_group(exp, table)
    det_group = canonical_group(det, table)

    if exp_group & det_group:
        return True

    if mode != "semantic":
        return False

    # tier 3: lemma equality
    return lemmatize(exp) == lemmatize(det)
