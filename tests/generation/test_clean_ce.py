from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from text2motion.app.config import GeneratorConfig, GeneratorTrainingConfig, TokenizerConfig
from text2motion.generation.contracts import Backbone, GeneratorTrainingRequest
from text2motion.generation.nn_model import MotionTokenGenerator
from text2motion.generation.trainer import (
    CleanCrossEntropyLoaders,
    GeneratorTrainer,
    train_generator,
)
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


def __noisy_trainer() -> GeneratorTrainer:
    torch.manual_seed(2026)
    trainer = GeneratorTrainer(
        MotionTokenGenerator(GEN),
        FiniteScalarQuantizationTokenizer(TOK),
        GeneratorTrainingConfig(lr=3e-3, cfg_dropout=0.5, pkeep=0.5),
        downsample=4,
    )
    captions = {"walk": torch.randn(GEN.d_text), "run": torch.randn(GEN.d_text)}
    trainer.encode = lambda texts: torch.stack([captions[text] for text in texts])
    return trainer


def __batches() -> list[SimpleNamespace]:
    generator = torch.Generator().manual_seed(7)
    return [
        SimpleNamespace(
            features=torch.randn(2, 32, 263, generator=generator),
            lengths=torch.tensor([32, 28]),
            captions=("walk", "run"),
        )
        for _ in range(3)
    ]


def test_the_clean_ce_repeats_and_ignores_the_training_noise():
    trainer = __noisy_trainer()
    batches = __batches()
    first = trainer.teacher_forced_ce(batches, "cpu")
    second = trainer.teacher_forced_ce(batches, "cpu")
    assert first == second
    assert first > 0.0


def test_the_clean_ce_changes_no_weight_and_restores_the_train_mode():
    trainer = __noisy_trainer()
    before = [parameter.detach().clone() for parameter in trainer.token_generator.parameters()]
    trainer.token_generator.train()
    trainer.teacher_forced_ce(__batches(), "cpu", max_batches=1)
    after = list(trainer.token_generator.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert trainer.token_generator.training


def test_the_clean_ce_fails_loud_on_an_empty_loader():
    with pytest.raises(ValueError, match="no batch"):
        __noisy_trainer().teacher_forced_ce([], "cpu")


class __CurveTrainer:
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])

    def __init__(self, validation_curve: list[float]) -> None:
        self.validation_curve = list(validation_curve)
        self.epochs = 0

    def train_epoch(self, loader, device, heartbeat=None):
        self.epochs += 1
        return {"ce": 1.0, "ric": 0.1, "rot6d": 0.1, "foot": 0.1, "total": 1.3}

    def teacher_forced_ce(self, loader, device, max_batches=0):
        if loader == "validation":
            return self.validation_curve[self.epochs - 1]
        return 1.0

    @contextmanager
    def ema_weights(self):
        yield None

    def resume_state(self, epoch, best_fid, stale, run_dir):
        return {"epoch": epoch, "best_fid": best_fid}


def test_the_run_stops_after_the_patience_without_a_lower_validation_ce(tmp_path):
    curve = __CurveTrainer([3.0, 2.5, 2.6, 2.7, 2.4, 2.3])
    records = []
    train_generator(
        cast("GeneratorTrainer", curve),
        SimpleNamespace(dataset=SimpleNamespace(set_epoch=lambda epoch: None)),
        GeneratorTrainingRequest(epochs=6, eval_every=100, val_ce_patience=2),
        device="cpu",
        evaluate=lambda split: {},
        save_best=lambda epoch, fid: None,
        resume_path=tmp_path / "last.pt",
        on_metrics=records.append,
        ce_loaders=CleanCrossEntropyLoaders(train="train", validation="validation"),
    )
    assert curve.epochs == 4
    gaps = [record["ce_gap"] for record in records if "ce_gap" in record]
    assert gaps == pytest.approx([2.0, 1.5, 1.6, 1.7])


def test_a_validation_ce_patience_needs_the_ce_loaders(tmp_path):
    with pytest.raises(ValueError, match="clean CE loaders"):
        train_generator(
            cast("GeneratorTrainer", __CurveTrainer([1.0])),
            None,
            GeneratorTrainingRequest(epochs=1, val_ce_patience=2),
            device="cpu",
            evaluate=lambda split: {},
            save_best=lambda epoch, fid: None,
            resume_path=tmp_path / "last.pt",
        )
