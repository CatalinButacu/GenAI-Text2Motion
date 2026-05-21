from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from .vocab_actions import ACTIONS, ActionDefinition

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
            self.texts.append(f"{action.name} {' '.join(action.keywords[:6])}")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.built = True
        log.info("[SemanticActionResolver] built embeddings for %d actions", len(self.keys))

    def resolve(self, text: str, threshold: float = 0.45) -> ActionDefinition | None:
        self.build()

        if self.model is None or self.embeddings is None:
            return None

        vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        sims = (self.embeddings @ vec.T).ravel()
        idx = int(sims.argmax())

        if sims[idx] >= threshold:
            matched = self.keys[idx]
            log.debug("[SemanticActionResolver] %r -> %r (sim=%.3f)", text, matched, sims[idx])

            return ACTIONS[matched]

        return None
