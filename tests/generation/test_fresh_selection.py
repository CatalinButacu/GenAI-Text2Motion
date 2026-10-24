from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from text2motion.generation.contracts import GeneratorTrainingRequest
from text2motion.generation.trainer import (
    GENERATOR_RESUME_FORMAT,
    GeneratorTrainer,
    train_generator,
)
from text2motion.motion.storage import RESUME_VERSION


class _FakeTrainer:
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])

    def __init__(self) -> None:
        self.saved_states = []

    def load_resume_state(self, state, legacy, extend_schedule):
        return state["epoch"] + 1, state["best_fid"], 0

    def train_epoch(self, loader, device, heartbeat=None):
        return {"ce": 1.0, "ric": 0.1, "rot6d": 0.1, "foot": 0.1, "total": 1.3}

    @contextmanager
    def ema_weights(self):
        yield None

    def resume_state(self, epoch, best_fid, stale):
        return {"epoch": epoch, "best_fid": best_fid}


def __run(
    tmp_path,
    fresh_selection: bool,
    write_resume: bool = True,
    fid: float = 0.30,
    saved: list | None = None,
):
    resume_path = tmp_path / "generator_last.pt"
    if write_resume:
        torch.save(
            {
                "format": GENERATOR_RESUME_FORMAT,
                "version": RESUME_VERSION,
                "epoch": 59,
                "best_fid": 0.18,
            },
            resume_path,
        )
    saved = [] if saved is None else saved
    request = GeneratorTrainingRequest(
        epochs=62, eval_every=1, resume=True, fresh_selection=fresh_selection
    )
    metrics = {
        "clips": 1,
        "fid": fid,
        "r_top1": 0.4,
        "r_top3": 0.7,
        "mm_dist": 3.7,
        "diversity": 9.0,
    }
    loader = SimpleNamespace(dataset=SimpleNamespace(set_epoch=lambda epoch: None))
    train_generator(
        cast(GeneratorTrainer, _FakeTrainer()),
        loader,
        request,
        "cpu",
        evaluate=lambda split: dict(metrics),
        save_best=lambda epoch, fid: saved.append((epoch, fid)),
        resume_path=resume_path,
    )
    return saved


def test_fresh_selection_saves_the_fine_tunes_own_best(tmp_path):
    assert __run(tmp_path, fresh_selection=False) == []
    assert __run(tmp_path, fresh_selection=True) == [(61, 0.30)]


def test_fresh_selection_refuses_to_start_without_the_resume_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        __run(tmp_path, fresh_selection=True, write_resume=False)


def test_a_non_finite_fid_stops_the_run_without_a_save(tmp_path):
    saved = []
    with pytest.raises(RuntimeError, match="not finite"):
        __run(tmp_path, fresh_selection=True, fid=float("nan"), saved=saved)

    assert saved == []
    assert torch.load(tmp_path / "generator_last.pt")["epoch"] == 59
