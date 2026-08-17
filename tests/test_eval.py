from pathlib import Path

import numpy as np
import pytest
import torch

from text2motion.app.config import load_config
from text2motion.evaluation.metrics import diversity, fid, mm_dist, r_precision


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

    from text2motion.evaluation.matcher import load_matchers

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


def test_fid_rejects_degenerate_input():
    import pytest

    from text2motion.evaluation.metrics import fid as _fid

    with pytest.raises(ValueError, match="at least|>= 2|needs"):
        _fid(np.zeros((1, 8), np.float32), np.zeros((5, 8), np.float32))


def test_fid_warns_when_covariance_is_rank_deficient(capsys):
    from text2motion.evaluation.metrics import fid as _fid

    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 64)).astype(np.float32)
    b = rng.normal(size=(20, 64)).astype(np.float32)

    _fid(a, b)
    assert "rank-deficient" in capsys.readouterr().out

    big_a = rng.normal(size=(80, 8)).astype(np.float32)
    big_b = rng.normal(size=(80, 8)).astype(np.float32)
    _fid(big_a, big_b)
    assert "rank-deficient" not in capsys.readouterr().out


def test_bootstrap_fid_reports_a_usable_interval():
    from text2motion.evaluation.metrics import bootstrap_fid

    rng = np.random.default_rng(0)
    real = rng.normal(size=(120, 16)).astype(np.float32)
    gen = real * 0.9 + rng.normal(size=(120, 16)).astype(np.float32) * 0.4

    out = bootstrap_fid(real, gen, resamples=40)

    assert out["fid_resamples"] == 40
    assert out["fid_std"] > 0
    assert out["fid_ci_lo"] < out["fid_boot_mean"] < out["fid_ci_hi"]


def test_paired_delta_separates_a_real_gap_and_not_an_identical_pair():
    from text2motion.evaluation.metrics import paired_fid_delta

    rng = np.random.default_rng(0)
    real = rng.normal(size=(150, 16)).astype(np.float32)
    close = real * 0.95 + rng.normal(size=(150, 16)).astype(np.float32) * 0.20
    far = real * 0.70 + rng.normal(size=(150, 16)).astype(np.float32) * 0.90

    gap = paired_fid_delta(real, close, far, resamples=40)
    assert gap["delta_mean"] < 0
    assert gap["separates"]

    same = paired_fid_delta(real, close, close, resamples=40)
    assert abs(same["delta_mean"]) < 1e-9
    assert not same["separates"]


def test_paired_delta_requires_aligned_clips():
    import pytest

    from text2motion.evaluation.metrics import paired_fid_delta

    rng = np.random.default_rng(0)
    real = rng.normal(size=(20, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="same clips"):
        paired_fid_delta(real, real, rng.normal(size=(19, 8)).astype(np.float32), resamples=5)
