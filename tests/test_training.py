import numpy as np
import torch

from text2motion.app.config import GeneratorConfig, TokenizerConfig, TrainingConfig
from text2motion.generation.losses import LossWeights, soft_decode
from text2motion.generation.model import MotionGeneratorModule
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.tokenization.model import ResidualFsqTokenizer
from text2motion.tokenization.trainer import Ema

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
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
    gen = MotionGeneratorModule(GEN)
    tokens = torch.randint(0, GEN.codebook_size, (2, 8, GEN.num_codebooks))
    logits = gen(tokens, torch.randn(2, GEN.d_text))
    recon = soft_decode(logits, tok)

    assert recon.shape == (2, 32, 263)  # decoder upsamples T'=8 -> T=32
    recon.abs().mean().backward()  # gradient flows through frozen decoder back to the generator
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gen.parameters())


def test_train_step_learns():
    torch.manual_seed(0)
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGeneratorModule(GEN)
    trainer = GeneratorTrainer(
        gen, tok, TrainingConfig(lr=3e-3, cfg_dropout=0.0, pkeep=1.0), downsample=4
    )
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
    gen = MotionGeneratorModule(GEN)
    trainer = GeneratorTrainer(
        gen, tok, TrainingConfig(cfg_dropout=1.0), downsample=4
    )  # always drop text
    parts = trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))

    assert all(v == v for v in parts.values())  # no NaNs


def test_term_split_reports_each_group():
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGeneratorModule(GEN)
    trainer = GeneratorTrainer(gen, tok, TrainingConfig(cfg_dropout=0.0, pkeep=1.0), downsample=4)
    parts = trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))

    for term in ("ce", "root", "ric", "rot6d", "vel", "foot", "total"):
        assert term in parts and parts[term] == parts[term]


def test_fk_consistency_terms_finite():
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGeneratorModule(GEN)
    cfg = TrainingConfig(
        cfg_dropout=0.0, pkeep=1.0, loss_weights=LossWeights(fk_self=0.5, fk_gt=0.5)
    )
    mean = np.zeros(263, np.float32)
    std = np.ones(263, np.float32)
    trainer = GeneratorTrainer(gen, tok, cfg, mean=mean, std=std, downsample=4)
    parts = trainer.train_step(
        torch.randn(2, 32, 263), torch.randn(2, GEN.d_text), torch.tensor([32, 24])
    )

    for term in ("fk_self", "fk_gt"):
        assert term in parts and parts[term] == parts[term]


def test_uncertainty_weighting_optimizes_log_vars():
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGeneratorModule(GEN)
    trainer = GeneratorTrainer(
        gen,
        tok,
        TrainingConfig(cfg_dropout=0.0, pkeep=1.0, loss_weighting="uncertainty"),
        downsample=4,
    )
    assert trainer.weighter is not None

    trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))
    grad = trainer.weighter.log_vars.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_ema_tracks_and_restores():
    gen = MotionGeneratorModule(GEN)
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


def test_decay_groups_exempt_norms_biases_and_the_ssm_memory_params():
    from dataclasses import replace as _replace

    from text2motion.app.config import GeneratorConfig
    from text2motion.generation.trainer import partition_parameters

    gen_cfg = GeneratorConfig(num_codebooks=2, codebook_size=64, d_model=64, n_layers=2)
    mamba = MotionGeneratorModule(_replace(gen_cfg, backbone="mamba"))
    transformer = MotionGeneratorModule(_replace(gen_cfg, backbone="transformer"))

    for model in (mamba, transformer):
        partition = partition_parameters(model)
        decay, no_decay = partition.decay, partition.no_decay
        assert decay and no_decay
        assert all(p.ndim >= 2 for p in decay)
        decayed_ids = {id(p) for p in decay}
        for name, param in model.named_parameters():
            if name.endswith("a_log") or name.endswith("d_skip"):
                assert id(param) not in decayed_ids, f"{name} must not be weight-decayed"

    m_names = {n for n, _ in mamba.named_parameters()}
    assert any(n.endswith("a_log") for n in m_names)
    assert not any(n.endswith("a_log") for n, _ in transformer.named_parameters())


def test_pretraining_optimizer_groups_do_not_decay_ssm_memory_params():
    from dataclasses import replace as _replace

    from text2motion.generation.trainer import adamw_groups

    mamba = MotionGeneratorModule(_replace(GEN, backbone="mamba"))
    cfg = TrainingConfig(weight_decay=0.01, decay_groups=True)
    opt = torch.optim.AdamW(
        adamw_groups(mamba, cfg.lr, cfg.decay_groups),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    decay_by_id = {
        id(param): float(group["weight_decay"])
        for group in opt.param_groups
        for param in group["params"]
    }

    assert len(decay_by_id) == len(list(mamba.parameters()))
    for name, param in mamba.named_parameters():
        if name.endswith(("a_log", "d_skip")):
            assert decay_by_id[id(param)] == 0.0, (
                f"{name} must not be weight-decayed in pretraining"
            )
