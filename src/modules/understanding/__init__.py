from __future__ import annotations

import logging

from src.utils.mem_profile import tracemalloc_snapshot

from .config import ParserConfig, SpacyConfig
from .models import ParsedAction, ParsedEntity, ParsedScene
from .spacy import SpacyParser

log = logging.getLogger(__name__)

PARSER: SpacyParser | None = None


def invoke(prompt: str, config: ParserConfig | None = None) -> ParsedScene:
    """M1: parse text into entities/actions via cached SpacyParser."""
    global PARSER

    if PARSER is None:
        PARSER = SpacyParser(config)

    with tracemalloc_snapshot("M1 parse"):
        parsed = PARSER.parse(prompt)

    log.info("[M1] %d entities, %d actions", len(parsed.entities), len(parsed.actions))

    return parsed


__all__ = ["invoke", "ParserConfig", "SpacyConfig", "ParsedScene", "ParsedEntity", "ParsedAction"]
