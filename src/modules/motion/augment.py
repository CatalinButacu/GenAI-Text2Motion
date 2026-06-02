"""SBERT-based prompt augmentation: reranker (US-08) + retriever (US-09)."""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.shared.constants import CONSTS

from .models import MotionClip

log = logging.getLogger(__name__)

SBERT_MODEL = "all-MiniLM-L6-v2"
INDEX_PROMPTS_KEY = "prompts"
INDEX_EMBEDDINGS_KEY = "embeddings"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class MotionReranker:
    """Rank a list of MotionClip candidates against a text prompt using SBERT."""

    def __init__(self, sbert_model: str = SBERT_MODEL) -> None:
        self.encoder = SentenceTransformer(sbert_model)

    def score_candidates(
        self, prompt: str, candidates: list[MotionClip]
    ) -> list[tuple[float, MotionClip]]:
        """Compute SBERT cosine scores for each candidate (empty list -> empty result)."""
        if not candidates:
            return []

        prompt_emb = self.encoder.encode(prompt, convert_to_numpy=True).astype(np.float32)
        descriptions = [build_candidate_description(c) for c in candidates]
        cand_embs = self.encoder.encode(descriptions, convert_to_numpy=True).astype(np.float32)

        return [
            (cosine_similarity(prompt_emb, emb), clip) for emb, clip in zip(cand_embs, candidates)
        ]

    def rank(self, prompt: str, candidates: list[MotionClip]) -> MotionClip:
        """Return the SBERT-best candidate from a non-empty list."""
        if not candidates:
            raise ValueError("candidates list must not be empty")

        if len(candidates) == 1:
            return candidates[0]

        scored = self.score_candidates(prompt, candidates)
        best = max(scored, key=lambda sc: sc[0])
        log.debug(
            "Reranker prompt=%r -> %d candidates, best=%.4f",
            prompt[:50],
            len(scored),
            best[0],
        )

        return best[1]

    def rank_all(self, prompt: str, candidates: list[MotionClip]) -> list[tuple[float, MotionClip]]:
        """Return all candidates sorted by score (descending) as (score, clip) tuples."""
        return sorted(self.score_candidates(prompt, candidates), key=lambda x: x[0], reverse=True)


def build_candidate_description(clip: MotionClip) -> str:
    """Build a short text description from clip metadata for SBERT scoring."""
    parts: list[str] = []

    if clip.action:
        parts.append(clip.action)

    if clip.smplx_params is not None:
        frames = len(clip.smplx_params)
        duration_sec = frames / CONSTS.runtime.motion_fps
        parts.append(f"{duration_sec:.1f} seconds of motion")

    return " ".join(parts) if parts else "motion clip"


class MotionRetriever:
    """SBERT + brute-force cosine nearest-neighbour retrieval over motion prompts.

    FAISS is not required; retrieval is done via matrix multiply on CPU which is
    fast enough for the index sizes in this project (< 100 k prompts).
    """

    def __init__(
        self,
        index_path: str,
        top_k: int = 3,
        sbert_model: str = SBERT_MODEL,
    ) -> None:
        self.top_k = top_k
        self.encoder = SentenceTransformer(sbert_model)
        data = np.load(index_path, allow_pickle=True)
        self.prompts: list[str] = list(data[INDEX_PROMPTS_KEY])
        embeddings = data[INDEX_EMBEDDINGS_KEY].astype(np.float32)
        # Normalise rows for fast cosine via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.embeddings: np.ndarray = embeddings / norms
        log.info("MotionRetriever loaded %d prompts from %s", len(self.prompts), index_path)

    def retrieve(self, prompt: str) -> list[str]:
        """Return up to top_k retrieved prompts (excludes the query itself if present)."""
        query_emb = self.encoder.encode(prompt, convert_to_numpy=True).astype(np.float32)
        norm = float(np.linalg.norm(query_emb))

        if norm > 0:
            query_emb = query_emb / norm

        scores = self.embeddings @ query_emb
        top_idxs = np.argsort(-scores)

        results: list[str] = []

        for idx in top_idxs:
            candidate = self.prompts[int(idx)]

            if candidate.strip().lower() == prompt.strip().lower():
                continue

            results.append(candidate)

            if len(results) >= self.top_k:
                break

        return results

    def augment_prompt(self, prompt: str) -> str:
        """Append retrieved prompts in brackets: 'a person walks [retrieved: jog, stride]'."""
        retrieved = self.retrieve(prompt)

        if not retrieved:
            return prompt

        context = ", ".join(retrieved)
        return f"{prompt} [retrieved: {context}]"

    @classmethod
    def build_index(
        cls,
        prompt_list: list[str],
        output_path: str,
        sbert_model: str = SBERT_MODEL,
    ) -> None:
        """Embed prompt_list with SBERT and save .npz at output_path."""
        if not prompt_list:
            raise ValueError("prompt_list must not be empty")

        encoder = SentenceTransformer(sbert_model)
        log.info("Building retrieval index for %d prompts ...", len(prompt_list))
        embeddings = encoder.encode(prompt_list, convert_to_numpy=True, show_progress_bar=True)
        np.savez(
            output_path,
            **{INDEX_PROMPTS_KEY: np.array(prompt_list), INDEX_EMBEDDINGS_KEY: embeddings},
        )
        log.info("Retrieval index saved to %s", output_path)
