import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from text2motion.app.config import ResidualVectorQuantizationConfig, TokenizerConfig
from text2motion.tokenization.ema import ExponentialMovingAverage
from text2motion.tokenization.nn_tokenizer import (
    FiniteScalarQuantizationTokenizer,
    ResidualVectorQuantizationTokenizer,
)
from text2motion.tokenization.tokenizer_trainer import TokenizerTrainer

FSQ = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
RVQ = ResidualVectorQuantizationConfig(
    in_dim=263,
    width=32,
    downsample=4,
    n_resblocks=1,
    num_quantizers=2,
    codebook_size=16,
    code_dim=8,
    quant_dropout=0.0,
)
NETWORKS = MappingProxyType(
    {
        "fsq": (lambda: FiniteScalarQuantizationTokenizer(FSQ), 0.0),
        "rvq": (lambda: ResidualVectorQuantizationTokenizer(RVQ), RVQ.commitment_beta),
    }
)


def load(monkeypatch, name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def factory(kind: str):
    build, commitment = NETWORKS[kind]
    return lambda: TokenizerTrainer(
        build(), lr=3e-3, weight_decay=0.0, ema_decay=0.9, commitment_beta=commitment
    )


def random_batch() -> torch.Tensor:
    return torch.randn(2, 32, 263, generator=torch.Generator().manual_seed(2026))


@pytest.mark.parametrize("kind", ["fsq", "rvq"])
def test_every_check_passes_on_a_small_network(monkeypatch, kind):
    preflight = load(monkeypatch, "tokenizer_preflight", "scripts/training/tokenizer_preflight.py")
    torch.manual_seed(2026)
    make_trainer = factory(kind)
    trainer = make_trainer()
    batch = random_batch()
    assert preflight.overfit_gate(trainer, batch, 200)["passed"]
    assert preflight.ema_coverage(trainer)["passed"]
    assert preflight.export_coherence(trainer, batch, make_trainer)["passed"]
    assert preflight.resume_round_trip(trainer, batch, make_trainer)["passed"]


def test_the_rvq_codebook_is_in_the_ema_and_differs_from_the_live_codebook(monkeypatch):
    preflight = load(monkeypatch, "tokenizer_preflight", "scripts/training/tokenizer_preflight.py")
    torch.manual_seed(2026)
    make_trainer = factory("rvq")
    trainer = make_trainer()
    batch = random_batch()
    preflight.overfit_gate(trainer, batch, 20)
    coverage = preflight.ema_coverage(trainer)
    assert any(name.endswith("embed") for name in coverage["floating_buffers"])
    assert preflight.export_coherence(trainer, batch, make_trainer)[
        "export_differs_from_live_weights"
    ]


def test_an_ema_without_buffers_fails_the_coverage_check(monkeypatch):
    preflight = load(monkeypatch, "tokenizer_preflight", "scripts/training/tokenizer_preflight.py")
    trainer = factory("rvq")()
    trainer.parameter_ema = ExponentialMovingAverage(trainer.tokenizer_network, 0.9)
    coverage = preflight.ema_coverage(trainer)
    assert not coverage["passed"]
    assert any(name.endswith("embed") for name in coverage["missing_from_ema"])


def test_the_challenger_guard_refuses_production_names_resume_and_existing_files(
    monkeypatch, tmp_path
):
    challenger = load(
        monkeypatch, "train_tokenizer_challenger", "scripts/training/train_tokenizer_challenger.py"
    )
    assert challenger.challenger_problems("fsq_g8_v1024.pt", False, tmp_path)
    assert challenger.challenger_problems(None, False, tmp_path)
    assert challenger.challenger_problems("rvq_l8_1024_c1b.pt", True, tmp_path)
    assert challenger.challenger_problems("rvq_l8_1024_c1b.pt", False, tmp_path) == []
    (tmp_path / "rvq_l8_1024_c1b_last.pt").write_bytes(b"")
    assert challenger.challenger_problems("rvq_l8_1024_c1b.pt", False, tmp_path)
    assert Path(tmp_path / "rvq_l8_1024_c1b.pt").exists() is False
