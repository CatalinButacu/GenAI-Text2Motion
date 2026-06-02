"""Tests for src/modules/motion/retrieval.py (US-09)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.modules.motion.augment import MotionRetriever


def buildFakeIndex(tmp_path: Path, prompts: list[str]) -> str:
    """Build a minimal .npz index with unit-vector embeddings for testing."""
    n = len(prompts)
    embeddings = np.eye(n, dtype=np.float32)  # each prompt gets a unique basis vector
    indexPath = str(tmp_path / "test_index.npz")
    np.savez(indexPath, prompts=np.array(prompts), embeddings=embeddings)
    return indexPath


class TestMotionRetrieverLoad:
    def test_loads_prompts(self, tmp_path):
        prompts = ["walk", "run", "jump"]
        indexPath = buildFakeIndex(tmp_path, prompts)

        with patch("src.modules.motion.augment.SentenceTransformer"):
            retriever = MotionRetriever(index_path=indexPath, top_k=2)

        assert len(retriever.prompts) == 3

    def test_embeddings_normalised(self, tmp_path):
        prompts = ["a", "b", "c"]
        indexPath = buildFakeIndex(tmp_path, prompts)

        with patch("src.modules.motion.augment.SentenceTransformer"):
            retriever = MotionRetriever(index_path=indexPath)

        norms = np.linalg.norm(retriever.embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(prompts)), atol=1e-5)


class TestRetrieve:
    def test_returns_topk(self, tmp_path):
        prompts = ["walk", "run", "jump", "sit", "dance"]
        indexPath = buildFakeIndex(tmp_path, prompts)

        with patch("src.modules.motion.augment.SentenceTransformer") as MockST:
            instance = MockST.return_value
            # Query embedding: aligned with "walk" (index 0)
            instance.encode.return_value = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            retriever = MotionRetriever(index_path=indexPath, top_k=2)
            results = retriever.retrieve("walk quickly")

        assert len(results) <= 2

    def test_excludes_exact_query_match(self, tmp_path):
        prompts = ["walk", "run", "jump"]
        indexPath = buildFakeIndex(tmp_path, prompts)

        with patch("src.modules.motion.augment.SentenceTransformer") as MockST:
            instance = MockST.return_value
            # Exact match with "walk"
            instance.encode.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            retriever = MotionRetriever(index_path=indexPath, top_k=2)
            results = retriever.retrieve("walk")

        assert "walk" not in results

    def test_returns_list_of_strings(self, tmp_path):
        prompts = ["a person walks", "a person runs"]
        indexPath = buildFakeIndex(tmp_path, prompts)

        with patch("src.modules.motion.augment.SentenceTransformer") as MockST:
            instance = MockST.return_value
            instance.encode.return_value = np.array([0.5, 0.5], dtype=np.float32)
            retriever = MotionRetriever(index_path=indexPath, top_k=1)
            results = retriever.retrieve("a person moves")

        assert all(isinstance(r, str) for r in results)


class TestAugmentPrompt:
    def test_augment_appends_retrieved(self, tmp_path):
        prompts = ["walk", "run", "jog"]
        indexPath = buildFakeIndex(tmp_path, prompts)

        with patch("src.modules.motion.augment.SentenceTransformer") as MockST:
            instance = MockST.return_value
            instance.encode.return_value = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            retriever = MotionRetriever(index_path=indexPath, top_k=2)
            result = retriever.augment_prompt("sprint")

        assert result.startswith("sprint")
        assert "[retrieved:" in result

    def test_augment_empty_retrieved_returns_original(self, tmp_path):
        """When all prompts are excluded (exact match), return original."""
        prompts = ["only_prompt"]
        indexPath = buildFakeIndex(tmp_path, prompts)

        with patch("src.modules.motion.augment.SentenceTransformer") as MockST:
            instance = MockST.return_value
            instance.encode.return_value = np.array([1.0], dtype=np.float32)
            retriever = MotionRetriever(index_path=indexPath, top_k=1)
            result = retriever.augment_prompt("only_prompt")

        assert result == "only_prompt"


class TestBuildIndex:
    def test_creates_npz_file(self, tmp_path):
        outputPath = str(tmp_path / "index.npz")

        with patch("src.modules.motion.augment.SentenceTransformer") as MockST:
            instance = MockST.return_value
            instance.encode.return_value = np.eye(3, dtype=np.float32)
            MotionRetriever.build_index(["a", "b", "c"], output_path=outputPath)

        assert Path(outputPath).exists()

    def test_empty_prompt_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            MotionRetriever.build_index([], output_path=str(tmp_path / "x.npz"))

    def test_saved_prompts_match(self, tmp_path):
        outputPath = str(tmp_path / "index.npz")
        prompts = ["walk forward", "jump high", "sit down"]

        with patch("src.modules.motion.augment.SentenceTransformer") as MockST:
            instance = MockST.return_value
            instance.encode.return_value = np.eye(3, dtype=np.float32)
            MotionRetriever.build_index(prompts, output_path=outputPath)

        data = np.load(outputPath, allow_pickle=True)
        saved = list(data["prompts"])
        assert saved == prompts
