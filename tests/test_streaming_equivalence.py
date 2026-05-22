"""Streaming inference must reproduce the parallel forward pass.

The chief correctness guarantee for the streaming PR: running
:meth:`TextToMotionSSM.stream_step` ``T'`` times must produce the same
codebook logits (within floating-point tolerance) as a single call to
:meth:`TextToMotionSSM.forward` on a parallel-scanned ``T'``-length window.

If this test fails, the streaming code is *not* an inference-time
refactor of the parallel path -- it computes a different function -- and
every downstream claim (constant-memory, TTFF, hidden-state carryover)
becomes untrue.

The model used here is intentionally tiny (d_model=64, n_layers=2,
short max_motion_length) so the test runs on CPU in well under a second.
"""

from __future__ import annotations

import unittest

import torch

from src.modules.motion.nn_models import TextToMotionSSM


class TinyCfg:
    """Minimal config for the streaming-equivalence smoke; CPU-only, no SBERT."""

    def __init__(self) -> None:
        self.d_model = 64
        self.d_state = 16
        self.n_layers = 2
        self.motion_dim = 168
        self.text_embed_dim = 64
        self.max_motion_length = 16
        self.max_text_length = 8
        self.vocab_size = 32
        self.bidirectional = False  # streaming requires unidirectional
        self.use_film = True
        self.use_sbert = False  # SimpleTextEncoder; no model download
        self.sbert_model = "all-MiniLM-L6-v2"
        self.freeze_sbert = True
        self.gradient_checkpointing = False
        self.arch = "independent"
        self.rvq_latent_dim = 16
        self.rvq_n_codebooks = 2
        self.rvq_codebook_size = 8
        self.rvq_down_t = 4


class TestStreamingMatchesParallel(unittest.TestCase):

    def test_stream_step_reproduces_forward(self):
        torch.manual_seed(0)
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).eval()
        # text input: integer token ids since use_sbert=False
        tokens = torch.tensor([[5, 7, 9, 11]], dtype=torch.long)

        with torch.no_grad():
            logits_par, len_par = model(tokens, motion_length=cfg.max_motion_length)

        # logits_par: (1, T', K, V) where T' = max_motion_length // rvq_down_t
        latent_len = cfg.max_motion_length // cfg.rvq_down_t
        self.assertEqual(logits_par.shape[1], latent_len)

        with torch.no_grad():
            state = model.stream_begin(tokens)
            streamed_logits = []
            last_len = None

            for _ in range(latent_len):
                logits_t, len_t, state = model.stream_step(state)
                streamed_logits.append(logits_t)
                last_len = len_t
        stream_cat = torch.cat(streamed_logits, dim=1)
        # Generous tolerance: per-layer FiLM + residual is the same arithmetic
        # but ordered differently (T at a time vs all at once). Float32 noise
        # accumulates across 2 layers and 4 steps.
        torch.testing.assert_close(stream_cat, logits_par, atol=5e-4, rtol=5e-4)
        # The decoder's length head reads the last latent step in the parallel
        # case, so the final stream_step's length_pred should agree.
        torch.testing.assert_close(last_len, len_par, atol=5e-4, rtol=5e-4)

    def test_stream_state_advances_and_caps(self):
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).eval()
        tokens = torch.tensor([[5, 7, 9, 11]], dtype=torch.long)
        latent_len = cfg.max_motion_length // cfg.rvq_down_t

        with torch.no_grad():
            state = model.stream_begin(tokens)
            self.assertEqual(state.latent_step, 0)
            self.assertEqual(state.max_steps, latent_len)

            for expected in range(1, latent_len + 1):
                _, _, state = model.stream_step(state)
                self.assertEqual(state.latent_step, expected)
            # past the end must raise -- caller should re-init or carry over
            with self.assertRaises(RuntimeError):
                model.stream_step(state)

    def test_stream_requires_unidirectional(self):
        cfg = TinyCfg()
        cfg.bidirectional = True
        model = TextToMotionSSM(cfg).eval()
        tokens = torch.tensor([[5, 7, 9, 11]], dtype=torch.long)

        with self.assertRaises(RuntimeError):
            model.stream_begin(tokens)

    def test_carry_over_preserves_hidden_state(self):
        """Action transitions reuse SSM state but reset the latent step counter."""
        cfg = TinyCfg()
        model = TextToMotionSSM(cfg).eval()
        t1 = torch.tensor([[5, 7, 9, 11]], dtype=torch.long)
        t2 = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

        with torch.no_grad():
            state = model.stream_begin(t1)

            for _ in range(2):
                _, _, state = model.stream_step(state)
            h_before = [h.clone() for h in state.layer_h]
            # carry_over for a new action: new cond, same hidden state
            new_cond = model.condition_proj(model.text_encoder(t2))
            state.carry_over(new_cond)
            self.assertEqual(state.latent_step, 0)

            for i, h in enumerate(state.layer_h):
                torch.testing.assert_close(h, h_before[i])


if __name__ == "__main__":
    unittest.main()
