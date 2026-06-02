from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.shared.constants import SEMANTIC_KEYWORDS_LIMIT, SEMANTIC_MATCH_THRESHOLD

from .actions import ACTIONS, ActionDefinition

log = logging.getLogger(__name__)


class SemanticActionResolver:
    instance: SemanticActionResolver | None = None

    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.keys: list[str] = []
        self.texts: list[str] = []
        self.built = False

    @classmethod
    def get_instance(cls) -> SemanticActionResolver:
        if cls.instance is None:
            cls.instance = cls()

        return cls.instance

    def build(self) -> None:
        if self.built:
            return

        for key, action in ACTIONS.items():
            self.keys.append(key)
            self.texts.append(
                f"{action.name} {' '.join(action.keywords[:SEMANTIC_KEYWORDS_LIMIT])}"
            )

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.built = True
        log.info("built embeddings for %d actions", len(self.keys))

    def resolve(
        self, text: str, threshold: float = SEMANTIC_MATCH_THRESHOLD
    ) -> ActionDefinition | None:
        self.build()

        if self.model is None or self.embeddings is None:
            return None

        vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        sims = (self.embeddings @ vec.T).ravel()
        idx = int(sims.argmax())

        if sims[idx] >= threshold:
            matched = self.keys[idx]
            log.debug("%r -> %r (sim=%.3f)", text, matched, sims[idx])

            return ACTIONS[matched]

        return None
