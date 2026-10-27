import numpy as np
import pytest
import torch

from text2motion.app.config import GeneratorConfig, GeneratorTrainingConfig, TokenizerConfig
from text2motion.generation.contracts import Backbone, LossWeighting
from text2motion.generation.losses import LossWeights, soft_decode
from text2motion.generation.nn_model import MotionTokenGenerator
from text2motion.generation.trainer import (
    GENERATOR_RESUME_FORMAT,
    GeneratorTrainer,
    check_schedule_horizon,
)
from text2motion.motion.storage import RESUME_VERSION, atomic_torch_save, is_legacy_resume
from text2motion.tokenization.ema import ExponentialMovingAverage
from text2motion.tokenization.nn_model import FiniteScalarQuantizationTokenizer

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
    backbone=Backbone.MAMBA,
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
    tok = FiniteScalarQuantizationTokenizer(TOK).eval().requires_grad_(False)
    gen = MotionTokenGenerator(GEN)
    tokens = torch.randint(0, GEN.codebook_size, (2, 8, GEN.num_codebooks))
    logits = gen(tokens, torch.randn(2, GEN.d_text))
    recon = soft_decode(logits, tok)

    assert recon.shape == (2, 32, 263)  # decoder upsamples T'=8 -> T=32
    recon.abs().mean().backward()  # gradient flows through frozen decoder back to the generator
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gen.parameters())


def test_train_step_learns():
    torch.manual_seed(0)
    tok = FiniteScalarQuantizationTokenizer(TOK)
    gen = MotionTokenGenerator(GEN)
    trainer = GeneratorTrainer(
        gen, tok, GeneratorTrainingConfig(lr=3e-3, cfg_dropout=0.0, pkeep=1.0), downsample=4
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
    tok = FiniteScalarQuantizationTokenizer(TOK)
    gen = MotionTokenGenerator(GEN)
    trainer = GeneratorTrainer(
        gen, tok, GeneratorTrainingConfig(cfg_dropout=1.0), downsample=4
    )  # always drop text
    parts = trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))

    assert all(v == v for v in parts.values())  # no NaNs


def test_term_split_reports_each_group():
    tok = FiniteScalarQuantizationTokenizer(TOK)
    gen = MotionTokenGenerator(GEN)
    trainer = GeneratorTrainer(
        gen, tok, GeneratorTrainingConfig(cfg_dropout=0.0, pkeep=1.0), downsample=4
    )
    parts = trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))

    for term in ("ce", "root", "ric", "rot6d", "vel", "foot", "total"):
        assert term in parts and parts[term] == parts[term]


def test_fk_consistency_terms_finite():
    tok = FiniteScalarQuantizationTokenizer(TOK)
    gen = MotionTokenGenerator(GEN)
    cfg = GeneratorTrainingConfig(
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
    tok = FiniteScalarQuantizationTokenizer(TOK)
    gen = MotionTokenGenerator(GEN)
    trainer = GeneratorTrainer(
        gen,
        tok,
        GeneratorTrainingConfig(
            cfg_dropout=0.0, pkeep=1.0, loss_weighting=LossWeighting.UNCERTAINTY
        ),
        downsample=4,
    )
    assert trainer.weighter is not None

    trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))
    grad = trainer.weighter.log_vars.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_ema_tracks_and_restores():
    gen = MotionTokenGenerator(GEN)
    ema = ExponentialMovingAverage(gen, decay=0.9)
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


def test_generator_step_with_a_nan_loss_raises_before_the_update():
    gen = MotionTokenGenerator(GEN)
    trainer = GeneratorTrainer(
        gen,
        FiniteScalarQuantizationTokenizer(TOK),
        GeneratorTrainingConfig(cfg_dropout=0.0, pkeep=1.0),
        downsample=4,
    )
    before = {name: value.clone() for name, value in gen.state_dict().items()}

    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.train_step(torch.randn(2, 32, 263), torch.full((2, GEN.d_text), float("nan")))

    for name, value in gen.state_dict().items():
        assert torch.equal(value, before[name])


def __resumable_trainer(grad_accum: int = 1) -> GeneratorTrainer:
    torch.manual_seed(0)
    config = GeneratorTrainingConfig(
        cfg_dropout=0.3,
        pkeep=0.8,
        grad_accum=grad_accum,
        loss_weighting=LossWeighting.UNCERTAINTY,
    )
    trainer = GeneratorTrainer(
        MotionTokenGenerator(GEN), FiniteScalarQuantizationTokenizer(TOK), config, downsample=4
    )
    trainer.build_scheduler(8)
    return trainer


def __train_one_epoch(trainer: GeneratorTrainer, motion: torch.Tensor, text: torch.Tensor) -> None:
    for _ in range(2):
        trainer.train_step(motion, text)
    trainer.apply_pending_gradients()


def test_a_resumed_run_continues_like_an_uninterrupted_run(tmp_path):
    motion = torch.randn(2, 32, 263, generator=torch.Generator().manual_seed(1))
    text = torch.randn(2, GEN.d_text, generator=torch.Generator().manual_seed(2))
    path = tmp_path / "generator_last.pt"
    uninterrupted = __resumable_trainer()
    __train_one_epoch(uninterrupted, motion, text)
    atomic_torch_save(uninterrupted.resume_state(0, 1.0, 0, None), path)
    __train_one_epoch(uninterrupted, motion, text)

    resumed = __resumable_trainer()
    resumed.load_resume_state(torch.load(path, weights_only=True), False, False)
    __train_one_epoch(resumed, motion, text)

    resumed_weights = resumed.token_generator.state_dict()
    for name, value in uninterrupted.token_generator.state_dict().items():
        assert torch.equal(value, resumed_weights[name]), name
    for name, shadow in uninterrupted.parameter_ema.shadow.items():
        assert torch.equal(shadow, resumed.parameter_ema.shadow[name]), name
    assert uninterrupted.weighter is not None
    assert resumed.weighter is not None
    assert torch.equal(uninterrupted.weighter.log_vars, resumed.weighter.log_vars)


def test_a_partial_accumulation_window_has_the_weight_of_a_full_window(monkeypatch):
    motion = torch.randn(2, 32, 263, generator=torch.Generator().manual_seed(1))
    text = torch.randn(2, GEN.d_text, generator=torch.Generator().manual_seed(2))
    trainers = []
    for grad_accum in (2, 4):
        torch.manual_seed(0)
        config = GeneratorTrainingConfig(
            cfg_dropout=0.0, pkeep=1.0, grad_accum=grad_accum, grad_clip=1e9
        )
        trainer = GeneratorTrainer(
            MotionTokenGenerator(GEN), FiniteScalarQuantizationTokenizer(TOK), config, downsample=4
        )
        monkeypatch.setattr(trainer.optimizer, "step", lambda: None)
        __train_one_epoch(trainer, motion, text)
        trainers.append(trainer)

    full_window, partial_window = trainers
    partial_parameters = dict(partial_window.token_generator.named_parameters())
    for name, parameter in full_window.token_generator.named_parameters():
        assert parameter.grad is not None
        partial_gradient = partial_parameters[name].grad
        assert partial_gradient is not None
        assert torch.allclose(parameter.grad, partial_gradient, atol=1e-6), name


def test_a_resume_state_with_pending_gradients_fails():
    trainer = __resumable_trainer(grad_accum=2)
    trainer.train_step(torch.randn(2, 32, 263), torch.randn(2, GEN.d_text))

    with pytest.raises(RuntimeError, match="pending gradients"):
        trainer.resume_state(0, 1.0, 0, None)


def test_an_old_resume_file_needs_the_legacy_flag(tmp_path):
    path = tmp_path / "generator_last.pt"
    with pytest.raises(ValueError, match="accept_legacy_resume"):
        is_legacy_resume({"epoch": 3}, path, GENERATOR_RESUME_FORMAT, False)
    with pytest.raises(ValueError, match="not 'text2motion.generator_resume'"):
        is_legacy_resume(
            {"format": "text2motion.tokenizer_resume", "version": RESUME_VERSION},
            path,
            GENERATOR_RESUME_FORMAT,
            True,
        )

    assert is_legacy_resume({"epoch": 3}, path, GENERATOR_RESUME_FORMAT, True)


def test_a_new_schedule_horizon_needs_the_extend_flag():
    with pytest.raises(ValueError, match="extend_schedule"):
        check_schedule_horizon(100, 160, extend=False)

    check_schedule_horizon(100, 160, extend=True)
    check_schedule_horizon(100, 100, extend=False)


def test_ema_update_after_resume_moves_the_loaded_shadow():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 2.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    ema.load_state_dict({"weight": torch.full((1, 1), 10.0)})
    ema.update(model)
    assert ema.shadow["weight"].item() == 6.0


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"weight": torch.zeros(1, 1), "bias": torch.zeros(1)},
        {"weight": torch.zeros(2, 1)},
        {"weight": torch.full((1, 1), float("nan"))},
    ],
    ids=["missing_key", "unexpected_key", "wrong_shape", "not_finite"],
)
def test_ema_rejects_a_bad_checkpoint_state(state):
    ema = ExponentialMovingAverage(torch.nn.Linear(1, 1, bias=False), decay=0.5)
    with pytest.raises((KeyError, ValueError)):
        ema.load_state_dict(state)


def test_decay_groups_exempt_norms_biases_and_the_ssm_memory_params():
    from dataclasses import replace as _replace

    from text2motion.app.config import GeneratorConfig
    from text2motion.generation.trainer import partition_parameters

    gen_cfg = GeneratorConfig(num_codebooks=2, codebook_size=64, d_model=64, n_layers=2)
    mamba = MotionTokenGenerator(_replace(gen_cfg, backbone=Backbone.MAMBA))
    transformer = MotionTokenGenerator(_replace(gen_cfg, backbone="transformer"))

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

    mamba = MotionTokenGenerator(_replace(GEN, backbone=Backbone.MAMBA))
    cfg = GeneratorTrainingConfig(weight_decay=0.01, decay_groups=True)
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
