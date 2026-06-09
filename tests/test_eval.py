"""Eval harness: metric math (non-slow) and the Guo matcher load contract (slow, needs finest.tar).

The slow test guards the `hidden`-buffer regression: finest.tar's motion/text encoders carry a
trained GRU initial state, and `load_matchers` must consume it (no silently-dropped checkpoint keys)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from text2motion.eval.metrics import diversity, fid, mm_dist, r_precision
from text2motion.shared.config import load_config


def test_metric_math():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((256, 512)).astype(np.float32)

    assert fid(feats, feats) < 1e-6
    assert fid(feats, feats + 5.0) > fid(feats, feats)
    assert mm_dist(feats, feats) < 1e-6
    assert diversity(feats) > 0

    aligned = r_precision(feats, feats, pool_size=32, top_k=3)
    assert aligned[0] > 0.99  # text==motion -> the true match always ranks first
    shuffled = r_precision(feats, rng.standard_normal((256, 512)).astype(np.float32))
    assert shuffled[0] < 0.2  # unrelated -> near chance (1/32)


@pytest.mark.slow
def test_matcher_loads_without_dropping_keys():
    cfg = load_config("configs/default.yaml")
    if not cfg.paths.eval_matcher or not Path(cfg.paths.eval_matcher).is_file():
        pytest.skip("finest.tar unavailable")

    from text2motion.eval.matcher import load_matchers

    motion, text = load_matchers(cfg.paths.eval_matcher, device="cpu")

    assert "hidden" in dict(motion.encoder.named_buffers())  # trained GRU init state, was dropped
    assert "hidden" in dict(text.named_buffers())

    feat = torch.randn(2, 40, 263)
    emb = motion(feat, torch.tensor([40, 30]))
    assert emb.shape == (2, 512) and torch.isfinite(emb).all()

    word_embs = torch.randn(2, 8, 300)
    pos_onehots = torch.zeros(2, 8, 15)
    temb = text(word_embs, pos_onehots, lengths=torch.tensor([8, 5]))
    assert temb.shape == (2, 512) and torch.isfinite(temb).all()
