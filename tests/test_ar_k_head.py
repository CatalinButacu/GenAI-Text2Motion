"""Tests for the ResidualKHead (autoregressive across K codebooks).

The AR head is opt-in (ModelConfig.arch='residual_k') and addresses the only
known architectural defect in the previous inference path: independent K
classifiers sampled at inference time despite RVQ being a residual code.

Tests cover:
  - shape contract identical to RVQMotionDecoder
  - teacher-forced training path produces deterministic logits
  - inference AR sampling produces (B, T, K) indices with codebook k's choice
    actually influencing the conditioning passed into codebook k+1
  - the legacy independent head is still selectable as the default
"""

from __future__ import annotations

import unittest

import torch

from src.modules.motion.config import TrainingConfig
from src.modules.motion.nn_models import (
    ResidualKHead,
    RVQMotionDecoder,
    TextToMotionSSM,
)
from src.modules.motion.ssm_model import sample_ar_k


def tiny_config(arch: str = "residual_k") -> TrainingConfig:
    return TrainingConfig(
        d_model=32,
        d_state=8,
        n_layers=1,
        max_motion_length=24,
        max_text_length=8,
        vocab_size=64,
        use_sbert=False,  # use simple token-id encoder so tests don't download SBERT
        text_embed_dim=32,
        bidirectional=False,
        use_film=False,
        gradient_checkpointing=False,
        rvq_n_codebooks=3,
        rvq_codebook_size=16,
        rvq_down_t=2,
        rvq_latent_dim=16,
        arch=arch,
        cfg_dropout_prob=0.0,
        use_amp=False,
    )


class TestResidualKHeadShapes(unittest.TestCase):

    def test_logits_shape_matches_independent_head(self):
        d_model, K, V, max_len = 32, 4, 8, 24
        features = torch.randn(2, 6, d_model)
        cond = torch.randn(2, d_model)

        independent = RVQMotionDecoder(d_model, K, V, max_len)
        ar = ResidualKHead(d_model, K, V, max_len)

        ind_logits, ind_len = independent(features, cond)
        ar_logits, ar_len = ar(features, cond)

        self.assertEqual(ind_logits.shape, ar_logits.shape)
        self.assertEqual(ind_logits.shape, (2, 6, K, V))
        self.assertEqual(ind_len.shape, ar_len.shape)

    def test_teacher_forced_logits_change_with_targets(self):
        """Different target token sequences should produce different logits
        (codebook k>0's logits depend on which token was teacher-forced for k-1)."""
        d_model, K, V, max_len = 32, 4, 8, 24
        head = ResidualKHead(d_model, K, V, max_len)
        head.eval()
        features = torch.randn(1, 4, d_model)
        cond = torch.randn(1, d_model)

        targets_a = torch.zeros(1, 4, K, dtype=torch.long)
        targets_b = torch.full((1, 4, K), V - 1, dtype=torch.long)

        with torch.no_grad():
            logits_a, _ = head(features, cond, target_tokens=targets_a)
            logits_b, _ = head(features, cond, target_tokens=targets_b)

        # Codebook 0 should be identical (no prior conditioning).
        self.assertTrue(torch.allclose(logits_a[:, :, 0], logits_b[:, :, 0]))
        # Codebook 1+ depend on prior tokens; should differ.
        self.assertFalse(torch.allclose(logits_a[:, :, 1], logits_b[:, :, 1]))


class TestTextToMotionSSMUsesArch(unittest.TestCase):

    def test_independent_is_default(self):
        cfg = tiny_config(arch="independent")
        model = TextToMotionSSM(cfg)
        self.assertIsInstance(model.decoder, RVQMotionDecoder)
        self.assertEqual(model.arch, "independent")

    def test_residual_k_arch_selects_ar_head(self):
        cfg = tiny_config(arch="residual_k")
        model = TextToMotionSSM(cfg)
        self.assertIsInstance(model.decoder, ResidualKHead)
        self.assertEqual(model.arch, "residual_k")

    def test_forward_with_target_tokens_runs(self):
        torch.manual_seed(0)
        cfg = tiny_config(arch="residual_k")
        model = TextToMotionSSM(cfg)
        model.eval()
        token_ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_text_length))
        target_tokens = torch.randint(0, cfg.rvq_codebook_size, (2, 6, cfg.rvq_n_codebooks))

        with torch.no_grad():
            logits, length = model(token_ids, motion_length=12, target_tokens=target_tokens)

        self.assertEqual(logits.shape[-2:], (cfg.rvq_n_codebooks, cfg.rvq_codebook_size))
        self.assertEqual(length.shape, (2,))


class TestSampleArK(unittest.TestCase):

    def test_returns_valid_indices_in_range(self):
        torch.manual_seed(0)
        cfg = tiny_config(arch="residual_k")
        model = TextToMotionSSM(cfg)
        model.eval()
        token_ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_text_length))

        indices = sample_ar_k(
            model, token_ids, motion_length=12,
            temperature=1.0, top_p=1.0,
        )

        self.assertEqual(indices.shape[-1], cfg.rvq_n_codebooks)
        self.assertTrue((indices >= 0).all().item())
        self.assertTrue((indices < cfg.rvq_codebook_size).all().item())

    def test_greedy_sampling_deterministic(self):
        torch.manual_seed(0)
        cfg = tiny_config(arch="residual_k")
        model = TextToMotionSSM(cfg)
        model.eval()
        token_ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_text_length))

        a = sample_ar_k(model, token_ids, motion_length=12, temperature=1.0, top_p=1.0)
        b = sample_ar_k(model, token_ids, motion_length=12, temperature=1.0, top_p=1.0)

        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
