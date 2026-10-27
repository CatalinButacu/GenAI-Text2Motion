from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from pydantic import ValidationError

from text2motion.app.config import TokenizerConfig
from text2motion.motion.storage import atomic_torch_save
from text2motion.tokenization.contracts import (
    DROPPED_CODE,
    ResidualVectorQuantizationConfig,
    TokenizerKind,
    TrainTokenizerRequest,
)
from text2motion.tokenization.nn_model import (
    FiniteScalarQuantizationTokenizer,
    build_tokenizer_network,
)
from text2motion.tokenization.trainer import TokenizerTrainer, train_tokenizer

CFG = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))


def __training_request(tmp_path, resume=False) -> TrainTokenizerRequest:
    return TrainTokenizerRequest(
        config=tmp_path / "unused.yaml",
        tokenizer=TokenizerKind.FINITE_SCALAR,
        epochs=1,
        eval_every=1,
        resume=resume,
    )


class _FakeTrainer:
    def train_epoch(self, loader, device):
        return {"recon": 1.0, "total": 1.0, "perplexity": 1.0, "usage_frac": 1.0, "commit": 0.0}

    @contextmanager
    def ema_weights(self):
        yield torch.nn.Linear(1, 1)

    def resume_state(self, epoch, best_fid, run_dir):
        return {"epoch": epoch, "best_fid": best_fid}


def __resumable_trainer() -> TokenizerTrainer:
    torch.manual_seed(0)
    return TokenizerTrainer(
        FiniteScalarQuantizationTokenizer(CFG), lr=1e-3, weight_decay=0.0, ema_decay=0.9
    )


def test_a_resumed_tokenizer_run_continues_like_an_uninterrupted_run(tmp_path):
    motion = torch.randn(2, 32, 263, generator=torch.Generator().manual_seed(1))
    path = tmp_path / "tokenizer_last.pt"
    uninterrupted = __resumable_trainer()
    uninterrupted.train_step(motion)
    atomic_torch_save(uninterrupted.resume_state(0, 1.0, None), path)
    uninterrupted.train_step(motion)
    uninterrupted.train_step(motion)

    resumed = __resumable_trainer()
    resumed.load_resume_state(torch.load(path, weights_only=True), False)
    resumed.train_step(motion)
    resumed.train_step(motion)

    resumed_weights = resumed.tokenizer_network.state_dict()
    for name, value in uninterrupted.tokenizer_network.state_dict().items():
        assert torch.equal(value, resumed_weights[name]), name
    for name, shadow in uninterrupted.parameter_ema.shadow.items():
        assert torch.equal(shadow, resumed.parameter_ema.shadow[name]), name


def test_tokenizer_step_with_a_nan_loss_raises_before_the_update():
    tokenizer = FiniteScalarQuantizationTokenizer(CFG)
    trainer = TokenizerTrainer(tokenizer, lr=1e-3, weight_decay=0.0, ema_decay=0.9)
    before = {name: value.clone() for name, value in tokenizer.state_dict().items()}
    motion = torch.randn(2, 32, 263)
    motion[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or Inf"):
        trainer.train_step(motion)

    for name, value in tokenizer.state_dict().items():
        assert torch.equal(value, before[name])


def test_a_non_finite_recon_fid_stops_the_run_without_a_save(tmp_path):
    checkpoint_path = tmp_path / "tokenizer.pt"
    resume_path = tmp_path / "tokenizer_last.pt"
    loader = SimpleNamespace(dataset=SimpleNamespace(set_epoch=lambda epoch: None))
    metrics = {"clips": 1, "mpjpe_mm": 1.0, "feature_l2": 1.0, "recon_fid": float("nan")}

    with pytest.raises(RuntimeError, match="not finite"):
        train_tokenizer(
            cast(TokenizerTrainer, _FakeTrainer()),
            loader,
            __training_request(tmp_path),
            "cpu",
            evaluate=lambda split: dict(metrics),
            checkpoint_path=checkpoint_path,
            resume_path=resume_path,
        )

    assert not checkpoint_path.exists()
    assert not resume_path.exists()


def test_resume_without_the_resume_file_fails(tmp_path):
    loader = SimpleNamespace(dataset=SimpleNamespace(set_epoch=lambda epoch: None))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        train_tokenizer(
            cast(TokenizerTrainer, _FakeTrainer()),
            loader,
            __training_request(tmp_path, resume=True),
            "cpu",
            evaluate=lambda split: {},
            checkpoint_path=tmp_path / "tokenizer.pt",
            resume_path=tmp_path / "tokenizer_last.pt",
        )


def test_code_statistics_match_a_count_per_codebook():
    trainer = __resumable_trainer()
    indices = torch.tensor([[[0, 1], [0, 2], [3, 2], [0, 2]]])
    perplexity, usage = trainer.code_statistics(indices)
    probabilities = torch.tensor([0.75, 0.25])
    expected = torch.exp(-(probabilities * probabilities.log()).sum())
    torch.testing.assert_close(perplexity, expected)
    torch.testing.assert_close(usage, torch.tensor(2 / trainer.tokenizer_network.codebook_size))


def test_code_statistics_skip_a_dropped_codebook():
    trainer = __resumable_trainer()
    indices = torch.tensor([[[0, DROPPED_CODE], [1, DROPPED_CODE]]])
    perplexity, usage = trainer.code_statistics(indices)
    torch.testing.assert_close(perplexity, torch.tensor(2.0))
    torch.testing.assert_close(usage, torch.tensor(2 / trainer.tokenizer_network.codebook_size))


def test_a_residual_vector_step_with_dropped_stages_gives_finite_values():
    torch.manual_seed(0)
    network = build_tokenizer_network(
        TokenizerConfig(kind=TokenizerKind.RESIDUAL_VECTOR),
        ResidualVectorQuantizationConfig(quant_dropout=1.0),
    )
    trainer = TokenizerTrainer(network, lr=1e-4, weight_decay=0.0, ema_decay=0.9)
    _, indices, _ = network(torch.randn(2, 16, 263))
    assert bool((indices == DROPPED_CODE).any())
    parts = trainer.train_step(torch.randn(2, 16, 263))
    assert all(torch.isfinite(value) for value in parts.values())


def test_a_non_finite_gradient_raises_before_the_update():
    trainer = __resumable_trainer()
    first = next(trainer.tokenizer_network.parameters())
    first.register_hook(lambda gradient: gradient * float("inf"))
    before = {name: value.clone() for name, value in trainer.tokenizer_network.state_dict().items()}
    shadow = {name: value.clone() for name, value in trainer.parameter_ema.shadow.items()}

    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.train_step(torch.randn(2, 32, 263))

    for name, value in trainer.tokenizer_network.state_dict().items():
        assert torch.equal(value, before[name]), name
    for name, value in trainer.parameter_ema.shadow.items():
        assert torch.equal(value, shadow[name]), name


def test_an_rvq_step_with_nan_motion_leaves_the_codebooks_finite():
    torch.manual_seed(0)
    network = build_tokenizer_network(
        TokenizerConfig(kind=TokenizerKind.RESIDUAL_VECTOR),
        ResidualVectorQuantizationConfig(width=32, code_dim=8, codebook_size=16, quant_dropout=0.0),
    )
    trainer = TokenizerTrainer(network, lr=1e-3, weight_decay=0.0, ema_decay=0.9)
    codebooks = {
        name: value.clone() for name, value in network.state_dict().items() if "embed" in name
    }
    motion = torch.randn(2, 16, 263)
    motion[1, 4, 9] = float("nan")

    with pytest.raises(ValueError, match="NaN or Inf"):
        trainer.train_step(motion)

    for name, value in codebooks.items():
        assert torch.equal(network.state_dict()[name], value), name


def test_an_epoch_without_a_batch_raises():
    with pytest.raises(ValueError, match="no batch"):
        __resumable_trainer().train_epoch([], "cpu")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 0),
        ("window", 0),
        ("batch_size", 0),
        ("eval_every", 0),
        ("max_eval_clips", 0),
        ("num_workers", -1),
        ("ema_decay", 1.5),
        ("ema_decay", -0.1),
    ],
)
def test_a_train_request_rejects_a_value_outside_its_range(tmp_path, field, value):
    with pytest.raises(ValidationError, match=field):
        TrainTokenizerRequest(
            config=tmp_path / "unused.yaml",
            tokenizer=TokenizerKind.FINITE_SCALAR,
            **{field: value},
        )
