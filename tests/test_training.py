import numpy as np
import torch

from text2motion.model.generator import MotionGenerator
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import GeneratorCfg, TokenizerCfg, TrainCfg
from text2motion.train.ema import Ema
from text2motion.train.losses import soft_decode
from text2motion.train.trainer import GeneratorTrainer

TOK = TokenizerCfg(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorCfg(
    backbone="mamba",
    d_model=64,
    n_layers=2,
    d_text=32,
    num_codebooks=2,
    codebook_size=16,
    max_seq_len=16,
    d_state=8,
    d_conv=4,
    expand=2,
    dt_rank=8,
)


def test_soft_decode_shape_and_grad():
    tok = ResidualFsqTokenizer(TOK).eval().requires_grad_(False)
    gen = MotionGenerator(GEN)
    tokens = torch.randint(0, GEN.codebook_size, (2, 8, GEN.num_codebooks))
    logits = gen(tokens, torch.randn(2, GEN.d_text))
    recon = soft_decode(logits, tok)

    assert recon.shape == (2, 32, 263)  # decoder upsamples T'=8 -> T=32
    recon.abs().mean().backward()  # gradient flows through frozen decoder back to the generator
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gen.parameters())


def test_train_step_learns():
    torch.manual_seed(0)
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGenerator(GEN)
    trainer = GeneratorTrainer(gen, tok, TrainCfg(lr=3e-3, cfg_dropout=0.0, pkeep=1.0))
    gt = torch.randn(2, 32, 263)
    text = torch.randn(2, GEN.d_text)

    first = trainer.train_step(gt, text)
    last = first

    for _ in range(150):
        last = trainer.train_step(gt, text)

    assert last["ce"] < 0.3 * first["ce"]  # generator memorises the (fixed) target tokens
    assert last["total"] < first["total"]
    assert all(not p.requires_grad for p in tok.parameters())


def test_cfg_dropout_step_is_finite():
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGenerator(GEN)
    trainer = GeneratorTrainer(gen, tok, TrainCfg(cfg_dropout=1.0))  # always drop text
    parts = trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))

    assert all(v == v for v in parts.values())  # no NaNs


def test_term_split_reports_each_group():
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGenerator(GEN)
    trainer = GeneratorTrainer(gen, tok, TrainCfg(cfg_dropout=0.0, pkeep=1.0))
    parts = trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))

    for term in ("ce", "root", "ric", "rot6d", "vel", "foot", "total"):
        assert term in parts and parts[term] == parts[term]


def test_fk_consistency_terms_finite():
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGenerator(GEN)
    cfg = TrainCfg(cfg_dropout=0.0, pkeep=1.0, w_fk_self=0.5, w_fk_gt=0.5)
    mean = np.zeros(263, np.float32)
    std = np.ones(263, np.float32)
    trainer = GeneratorTrainer(gen, tok, cfg, mean=mean, std=std)
    parts = trainer.train_step(
        torch.randn(2, 32, 263), torch.randn(2, GEN.d_text), torch.tensor([32, 24])
    )

    for term in ("fk_self", "fk_gt"):
        assert term in parts and parts[term] == parts[term]


def test_uncertainty_weighting_optimizes_log_vars():
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGenerator(GEN)
    trainer = GeneratorTrainer(
        gen, tok, TrainCfg(cfg_dropout=0.0, pkeep=1.0, loss_weighting="uncertainty")
    )
    assert trainer.weighter is not None

    trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))
    grad = trainer.weighter.log_vars.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_ema_tracks_and_restores():
    gen = MotionGenerator(GEN)
    ema = Ema(gen, decay=0.9)
    opt = torch.optim.SGD(gen.parameters(), lr=0.1)

    for _ in range(5):
        loss = (
            gen(torch.randint(0, GEN.codebook_size, (2, 8, GEN.num_codebooks)), torch.randn(2, 32))
            .pow(2)
            .mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(gen)

    name, param = next(iter(gen.named_parameters()))
    assert not torch.allclose(ema.shadow[name], param)  # EMA lags the live weights

    ema.copy_to(gen)
    assert torch.allclose(dict(gen.named_parameters())[name], ema.shadow[name])
    ema.restore(gen)
    assert torch.allclose(dict(gen.named_parameters())[name], param)
