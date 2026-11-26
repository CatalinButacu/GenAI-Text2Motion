import json
import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from text2motion.generation.contracts import Backbone, GeneratorConfig, GeneratorTrainingConfig
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.requests import GeneratorTrainingRequest
from text2motion.generation.training.epochs import train_generator
from text2motion.generation.training.resume import GENERATOR_RESUME_FORMAT
from text2motion.generation.training.selection import check_ce_stop_setup
from text2motion.generation.training.trainer import (
    CleanCrossEntropyLoaders,
    GeneratorTrainer,
)
from text2motion.motion.contracts import RunOutputs
from text2motion.motion.storage import RESUME_VERSION
from text2motion.tokenization.contracts import TokenizerConfig
from text2motion.tokenization.nn_tokenizer import FiniteScalarQuantizationTokenizer

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


class _CurveCheckpoint:
    def __init__(self, trainer) -> None:
        self.trainer = trainer

    def resume_state(self, epoch, best_fid, stale, run_dir):
        return self.trainer.resume_state(epoch, best_fid, stale, run_dir)

    def load_resume_state(self, state, legacy, extend_schedule):
        return self.trainer.load_resume_state(state, legacy, extend_schedule)


@pytest.fixture(autouse=True)
def curve_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "text2motion.generation.training.epochs.GeneratorCheckpoint", _CurveCheckpoint
    )
    monkeypatch.setattr(
        "text2motion.generation.training.resume.GeneratorCheckpoint", _CurveCheckpoint
    )


def noisy_trainer() -> GeneratorTrainer:
    torch.manual_seed(2026)
    trainer = GeneratorTrainer(
        MotionTokenGenerator(GEN),
        FiniteScalarQuantizationTokenizer(TOK),
        GeneratorTrainingConfig(lr=3e-3, cfg_dropout=0.5, pkeep=0.5),
        downsample=4,
    )
    captions = {"walk": torch.randn(GEN.d_text), "run": torch.randn(GEN.d_text)}
    trainer.batch.encode = lambda texts: torch.stack([captions[text] for text in texts])
    return trainer


def random_batches() -> list[SimpleNamespace]:
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
    trainer = noisy_trainer()
    batches = random_batches()
    first = trainer.teacher_forced_ce(batches, "cpu")
    second = trainer.teacher_forced_ce(batches, "cpu")
    assert first == second
    assert first > 0.0


def test_the_clean_ce_changes_no_weight_and_restores_the_train_mode():
    trainer = noisy_trainer()
    before = [parameter.detach().clone() for parameter in trainer.token_generator.parameters()]
    trainer.token_generator.train()
    trainer.teacher_forced_ce(random_batches(), "cpu", max_batches=1)
    after = list(trainer.token_generator.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert trainer.token_generator.training


def test_the_clean_ce_fails_loud_on_an_empty_loader():
    with pytest.raises(ValueError, match="no batch"):
        noisy_trainer().teacher_forced_ce([], "cpu")


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
        return {
            "format": GENERATOR_RESUME_FORMAT,
            "version": RESUME_VERSION,
            "epoch": epoch,
            "best_fid": best_fid,
            "stale_evaluations": stale,
            "run_dir": None if run_dir is None else str(run_dir),
        }

    def load_resume_state(self, state, legacy, extend_schedule):
        self.epochs = state["epoch"] + 1
        return self.epochs, state["best_fid"], state["stale_evaluations"]


def train_curve(tmp_path, curve, epochs, resume=False, fresh_selection=False, run_dir=None):
    records = []
    train_generator(
        cast("GeneratorTrainer", curve),
        SimpleNamespace(dataset=SimpleNamespace(set_epoch=lambda epoch: None)),
        GeneratorTrainingRequest(
            epochs=epochs, eval_every=100, resume=resume, fresh_selection=fresh_selection
        ),
        device="cpu",
        evaluate=lambda split: {
            "clips": 1,
            "fid": 1.0,
            "r_top1": 0.5,
            "r_top3": 0.7,
            "mm_dist": 1.0,
            "diversity": 1.0,
        },
        save_best=lambda epoch, fid: None,
        resume_path=tmp_path / "last.pt",
        outputs=RunOutputs(run_dir=run_dir, on_metrics=records.append),
        ce_loaders=CleanCrossEntropyLoaders(train="train", validation="validation"),
    )
    return records


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
        outputs=RunOutputs(on_metrics=records.append),
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


def test_default_stop_allows_five_retries_and_stops_on_the_sixth_tie(tmp_path):
    curve = __CurveTrainer([2.0] * 7 + [1.9])
    train_curve(tmp_path, curve, epochs=8)
    state = torch.load(tmp_path / "last.pt", weights_only=True)

    assert curve.epochs == 7
    assert state["clean_ce_progress"] == {"best": 2.0, "stale": 6}


def test_a_lower_validation_ce_resets_the_full_six_epoch_patience(tmp_path):
    curve = __CurveTrainer([2.0] * 6 + [1.9] * 7 + [1.8])
    train_curve(tmp_path, curve, epochs=14)

    assert curve.epochs == 13
    assert torch.load(tmp_path / "last.pt", weights_only=True)["clean_ce_progress"] == {
        "best": 1.9,
        "stale": 6,
    }


@pytest.mark.parametrize("old_checkpoint", [False, True])
def test_resume_preserves_the_streak_instead_of_granting_six_new_retries(tmp_path, old_checkpoint):
    values = [2.0] + [2.1] * 6 + [1.9]
    first = __CurveTrainer(values)
    folder = tmp_path / "run"
    folder.mkdir()
    records = train_curve(tmp_path, first, epochs=4, run_dir=folder)
    (folder / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in [*records, {"epoch": 5, "val_ce": 1.0}]),
        encoding="utf-8",
    )
    if old_checkpoint:
        state = torch.load(tmp_path / "last.pt", weights_only=True)
        del state["clean_ce_progress"]
        torch.save(state, tmp_path / "last.pt")
    resumed = __CurveTrainer(values)
    train_curve(tmp_path, resumed, epochs=8, resume=True, run_dir=folder)

    assert first.epochs == 4
    assert resumed.epochs == 7
    assert torch.load(tmp_path / "last.pt", weights_only=True)["clean_ce_progress"]["stale"] == 6


def test_an_exhausted_resume_budget_stops_before_another_training_epoch(tmp_path):
    values = [2.0] + [2.1] * 6 + [1.9]
    train_curve(tmp_path, __CurveTrainer(values), epochs=8)
    resumed = __CurveTrainer(values)
    assert train_curve(tmp_path, resumed, epochs=8, resume=True) == []
    assert resumed.epochs == 7


def test_old_resume_without_ce_history_fails_instead_of_resetting_patience(tmp_path):
    values = [2.0] + [2.1] * 6 + [1.9]
    train_curve(tmp_path, __CurveTrainer(values), epochs=4)
    state = torch.load(tmp_path / "last.pt", weights_only=True)
    del state["clean_ce_progress"]
    torch.save(state, tmp_path / "last.pt")
    with pytest.raises(ValueError, match="recorded run directory"):
        train_curve(tmp_path, __CurveTrainer(values), epochs=8, resume=True)


def test_fresh_selection_is_an_explicit_ce_patience_reset(tmp_path):
    values = [2.0] + [2.1] * 9 + [2.0]
    train_curve(tmp_path, __CurveTrainer(values), epochs=4)
    resumed = __CurveTrainer(values)
    train_curve(tmp_path, resumed, epochs=11, resume=True, fresh_selection=True)
    assert resumed.epochs == 11


@pytest.mark.parametrize(
    "progress",
    [
        {"best": 2.0, "stale": -1},
        {"best": float("nan"), "stale": 0},
        {"best": 2.0, "stale": 3.1},
    ],
)
def test_invalid_saved_ce_patience_fails_before_training(tmp_path, progress):
    values = [2.0] + [2.1] * 6 + [1.9]
    train_curve(tmp_path, __CurveTrainer(values), epochs=4)
    state = torch.load(tmp_path / "last.pt", weights_only=True)
    state["clean_ce_progress"] = progress
    torch.save(state, tmp_path / "last.pt")
    with pytest.raises(ValueError, match="invalid saved"):
        train_curve(tmp_path, __CurveTrainer(values), epochs=8, resume=True)


def test_a_planned_stop_ends_the_run_and_evaluates_its_last_epoch(tmp_path):
    curve = __CurveTrainer([3.0, 2.9, 2.8, 2.7, 2.6, 2.5])
    evaluated = []
    train_generator(
        cast("GeneratorTrainer", curve),
        SimpleNamespace(dataset=SimpleNamespace(set_epoch=lambda epoch: None)),
        GeneratorTrainingRequest(epochs=60, eval_every=100, stop_epoch=3),
        device="cpu",
        evaluate=lambda split: (
            evaluated.append(split)
            or {
                "clips": 1,
                "fid": 1.0,
                "r_top1": 0.5,
                "r_top3": 0.7,
                "mm_dist": 1.0,
                "diversity": 1.0,
            }
        ),
        save_best=lambda epoch, fid: None,
        resume_path=tmp_path / "last.pt",
        ce_loaders=CleanCrossEntropyLoaders(train="train", validation="validation"),
    )
    assert curve.epochs == 3
    assert len(evaluated) == 1


def test_a_stop_after_the_schedule_fails_before_training(tmp_path):
    with pytest.raises(ValueError, match="stop_epoch"):
        train_generator(
            cast("GeneratorTrainer", __CurveTrainer([1.0])),
            None,
            GeneratorTrainingRequest(epochs=2, stop_epoch=3),
            device="cpu",
            evaluate=lambda split: {},
            save_best=lambda epoch, fid: None,
            resume_path=tmp_path / "last.pt",
            ce_loaders=CleanCrossEntropyLoaders(train="train", validation="validation"),
        )


def test_whole_step_noise_replaces_all_codes_of_a_step_together(monkeypatch):
    torch.manual_seed(2026)
    trainer = GeneratorTrainer(
        MotionTokenGenerator(GEN),
        FiniteScalarQuantizationTokenizer(TOK),
        GeneratorTrainingConfig(cfg_dropout=0.0, pkeep=0.5, pkeep_whole_step=True),
        downsample=4,
    )
    inputs = []
    forward = trainer.token_generator.forward
    monkeypatch.setattr(
        trainer.token_generator,
        "forward",
        lambda tokens, text: inputs.append(tokens) or forward(tokens, text),
    )
    monkeypatch.setattr(torch, "randint_like", lambda tokens, high: (tokens + 1) % high)
    motion = torch.randn(4, 64, 263)
    text = torch.randn(4, GEN.d_text)
    trainer.batch_loss_parts(motion, text, None, with_training_noise=False)
    trainer.batch_loss_parts(motion, text, None, with_training_noise=True)
    changed = inputs[1] != inputs[0]

    assert torch.equal(changed.all(dim=-1), changed.any(dim=-1))
    assert 0 < changed.all(dim=-1).float().mean() < 1


def ce_with_text(cfg_dropout: float, text: torch.Tensor) -> float:
    torch.manual_seed(2026)
    trainer = GeneratorTrainer(
        MotionTokenGenerator(GEN),
        FiniteScalarQuantizationTokenizer(TOK),
        GeneratorTrainingConfig(cfg_dropout=cfg_dropout, pkeep=0.5),
        downsample=4,
    )
    trainer.batch.encode = lambda texts: text.expand(len(texts), -1)
    return trainer.teacher_forced_ce(random_batches(), "cpu")


def test_an_unconditional_run_measures_its_clean_ce_without_the_text():
    text = torch.randn(1, GEN.d_text, generator=torch.Generator().manual_seed(3))
    empty = torch.zeros(1, GEN.d_text)
    assert ce_with_text(1.0, text) == ce_with_text(1.0, empty)
    assert ce_with_text(0.5, text) != ce_with_text(0.5, empty)


def test_val_ce_selection_saves_at_each_lower_validation_ce_and_never_on_fid(tmp_path):
    saved = []
    train_generator(
        cast("GeneratorTrainer", __CurveTrainer([3.0, 2.5, 2.6, 2.4])),
        SimpleNamespace(dataset=SimpleNamespace(set_epoch=lambda epoch: None)),
        GeneratorTrainingRequest(epochs=4, eval_every=2, select_on_val_ce=True),
        device="cpu",
        evaluate=lambda split: {
            "clips": 1,
            "fid": 1.0,
            "r_top1": 0.5,
            "r_top3": 0.7,
            "mm_dist": 1.0,
            "diversity": 1.0,
        },
        save_best=lambda epoch, score: saved.append((epoch, score)),
        resume_path=tmp_path / "last.pt",
        ce_loaders=CleanCrossEntropyLoaders(train="train", validation="validation"),
    )
    assert [epoch for epoch, _ in saved] == [1, 2, 4]
    assert all(math.isnan(score) for _, score in saved)


def test_val_ce_selection_needs_the_clean_ce_loaders():
    request = GeneratorTrainingRequest(select_on_val_ce=True, val_ce_patience=0)
    with pytest.raises(ValueError, match="select_on_val_ce"):
        check_ce_stop_setup(request, None)
