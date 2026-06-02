"""Tests for src/modules/motion/reranker.py (US-08)."""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.modules.motion.models import MotionClip, MotionSource
from src.modules.motion.augment import (
    MotionReranker,
    build_candidate_description,
    cosine_similarity,
)


def makeClip(action: str = "walk", numFrames: int = 60) -> MotionClip:
    return MotionClip(
        action=action,
        smplx_params=np.zeros((numFrames, 168), dtype=np.float32),
    )


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0])
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_both_zero_returns_zero(self):
        a = np.array([0.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(0.0)


class TestBuildCandidateDescription:
    def test_includes_action(self):
        clip = makeClip(action="run")
        desc = build_candidate_description(clip)
        assert "run" in desc

    def test_includes_duration(self):
        clip = makeClip(numFrames=90)
        desc = build_candidate_description(clip)
        assert "3.0" in desc  # 90/30 = 3.0 seconds

    def test_fallback_for_empty_action(self):
        clip = makeClip(action="")
        desc = build_candidate_description(clip)
        assert isinstance(desc, str)
        assert len(desc) > 0


class TestMotionReranker:
    @pytest.fixture
    def mockEncoder(self):
        """Mock SentenceTransformer.encode to return distinguishable embeddings."""
        with patch("src.modules.motion.augment.SentenceTransformer") as MockST:
            instance = MockST.return_value

            def encodeFunc(texts, convert_to_numpy=True):
                if isinstance(texts, str):
                    # prompt embedding: unit vector along dim 0
                    return np.array([1.0, 0.0], dtype=np.float32)
                # candidate embeddings: first is aligned, rest are not
                embeddings = []
                for i, _ in enumerate(texts):
                    if i == 0:
                        embeddings.append([1.0, 0.0])
                    else:
                        embeddings.append([0.0, 1.0])
                return np.array(embeddings, dtype=np.float32)

            instance.encode.side_effect = encodeFunc
            yield instance

    def test_rank_single_candidate_returns_it(self, mockEncoder):
        reranker = MotionReranker()
        clip = makeClip()
        result = reranker.rank("walk forward", [clip])
        assert result is clip

    def test_rank_selects_best_candidate(self, mockEncoder):
        reranker = MotionReranker()
        clipA = makeClip(action="walk")
        clipB = makeClip(action="jump")
        # mockEncoder makes first candidate most similar
        result = reranker.rank("walk forward", [clipA, clipB])
        assert result is clipA

    def test_rank_empty_raises(self, mockEncoder):
        reranker = MotionReranker()
        with pytest.raises(ValueError, match="empty"):
            reranker.rank("walk", [])

    def test_rank_all_sorted_descending(self, mockEncoder):
        reranker = MotionReranker()
        clips = [makeClip(action=f"action_{i}") for i in range(3)]
        ranked = reranker.rank_all("prompt", clips)
        scores = [s for s, _ in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_all_empty_returns_empty(self, mockEncoder):
        reranker = MotionReranker()
        assert reranker.rank_all("prompt", []) == []
