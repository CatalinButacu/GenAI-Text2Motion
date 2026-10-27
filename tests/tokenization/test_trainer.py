from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from text2motion.app.config import TokenizerConfig
from text2motion.motion.storage import atomic_torch_save
from text2motion.tokenization.contracts import TokenizerTrainingRequest
from text2motion.tokenization.nn_model import FiniteScalarQuantizationTokenizer
from text2motion.tokenization.trainer import TokenizerTrainer, train_tokenizer

CFG = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))


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

    with pytest.raises(RuntimeError, match="non-finite"):
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
            TokenizerTrainingRequest(epochs=1, eval_every=1),
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
            TokenizerTrainingRequest(epochs=1, eval_every=1, resume=True),
            "cpu",
            evaluate=lambda split: {},
            checkpoint_path=tmp_path / "tokenizer.pt",
            resume_path=tmp_path / "tokenizer_last.pt",
        )
