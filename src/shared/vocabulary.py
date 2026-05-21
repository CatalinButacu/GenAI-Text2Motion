from __future__ import annotations

from .vocab_actions import ACTIONS, ActionCategory, ActionDefinition
from .vocab_objects import OBJECTS, ObjectCategory, ObjectDefinition
from .vocab_resolve import (
    get_action_by_keyword,
    get_object_by_keyword,
    register_nlp,
    resolve_action,
)
from .vocab_semantic import SemanticActionResolver

__all__ = [
    "ActionCategory",
    "ActionDefinition",
    "ACTIONS",
    "ObjectCategory",
    "ObjectDefinition",
    "OBJECTS",
    "register_nlp",
    "get_action_by_keyword",
    "get_object_by_keyword",
    "resolve_action",
    "SemanticActionResolver",
]
