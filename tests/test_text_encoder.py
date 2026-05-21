"""Tests for PretrainedTextEncoder (SBERT + CLIP via sentence-transformers).

These tests download models on first run (~80 MB for clip-ViT-B-32, ~90 MB
for all-MiniLM-L6-v2). Marked slow so they don't block fast CI; run with
``pytest -m slow tests/test_text_encoder.py`` when you want to verify CLIP
integration end-to-end.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from src.modules.motion.nn_models import PretrainedTextEncoder, SBERTTextEncoder


@pytest.mark.slow
class TestPretrainedTextEncoderSbert(unittest.TestCase):

    def test_loads_default_sbert_and_projects(self):
        encoder = PretrainedTextEncoder(d_model=128, model_name="all-MiniLM-L6-v2")
        # all-MiniLM-L6-v2 is 384-d; should be probed correctly.
        self.assertEqual(encoder.encoder_dim, 384)
        # Forward must produce (B, d_model).
        out = encoder(["a person walks forward", "someone jumps"])
        self.assertEqual(out.shape, (2, 128))

    def test_frozen_flag_disables_gradients(self):
        encoder = PretrainedTextEncoder(d_model=128, freeze=True)
        for p in encoder.sbert.parameters():
            self.assertFalse(p.requires_grad)
        # The projection head MUST stay trainable.
        for p in encoder.proj.parameters():
            self.assertTrue(p.requires_grad)


@pytest.mark.slow
class TestPretrainedTextEncoderClip(unittest.TestCase):

    def test_clip_b32_loads_with_correct_dim(self):
        encoder = PretrainedTextEncoder(d_model=128, model_name="clip-ViT-B-32")
        # clip-ViT-B-32 text encoder outputs 512-d.
        self.assertEqual(encoder.encoder_dim, 512)
        out = encoder(["a person walks forward"])
        self.assertEqual(out.shape, (1, 128))

    def test_clip_output_finite_and_different_per_prompt(self):
        encoder = PretrainedTextEncoder(d_model=64, model_name="clip-ViT-B-32")
        a = encoder(["a person walks"])
        b = encoder(["a person runs"])
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.isfinite(b).all())
        # Different verbs should produce different projected embeddings.
        self.assertFalse(torch.equal(a, b))


class TestBackwardCompatAlias(unittest.TestCase):

    def test_sbert_text_encoder_is_alias(self):
        """Old code that imported SBERTTextEncoder must still work."""
        self.assertIs(SBERTTextEncoder, PretrainedTextEncoder)


if __name__ == "__main__":
    unittest.main()
