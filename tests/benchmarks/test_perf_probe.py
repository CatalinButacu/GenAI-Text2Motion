from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts.benchmarks import perf_probe as probe

from text2motion.generation.contracts import Backbone
from text2motion.motion.representation import DIM


def test_precision_error_has_known_absolute_and_relative_values():
    measured = probe.precision_error(np.array([3.0, 4.0]), np.array([3.0, 5.0]))
    assert measured["max_abs"] == 1
    assert measured["relative_l2"] == pytest.approx(0.2)
    assert probe.precision_error(np.zeros(2), np.zeros(2))["relative_l2"] == 0


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (np.ones(2), np.ones(3)),
        (np.ones(2), np.full(2, np.nan)),
        (np.zeros(2), np.ones(2)),
        (np.zeros(0), np.zeros(0)),
    ],
)
def test_invalid_precision_comparisons_fail(reference, candidate):
    with pytest.raises(ValueError):
        probe.precision_error(reference, candidate)


def test_fixed_probe_batch_uses_seed2026_and_is_repeatable():
    first = probe.fixed_batch(2, 12, "cpu", 32)
    second = probe.fixed_batch(2, 12, "cpu", 32)
    generator = torch.Generator().manual_seed(2026)
    expected = torch.randn(2, 12, DIM, generator=generator)
    torch.testing.assert_close(first[0], expected, rtol=0, atol=0)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_precision_probe_requires_explicit_outputs():
    with pytest.raises(ValueError, match="out_path"):
        probe.precision_benchmark(Namespace(out_path=None, summary_csv=None))


def test_precision_probe_refuses_to_overwrite_evidence(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("{}")
    with pytest.raises(FileExistsError):
        probe.precision_benchmark(Namespace(out_path=path, summary_csv=tmp_path / "new.csv"))


def test_precision_cli_accepts_measured_output_paths():
    args = probe.build_parser().parse_args(
        [
            "--probe",
            "precision",
            "--out_path",
            "result.json",
            "--summary_csv",
            "result.csv",
            "--tokenizer_ckpt",
            r"runs\checkpoints\tokenizer\fsq_g8_v1024.pt",
        ]
    )
    assert args.probe == "precision" and args.out_path.name == "result.json"
    assert args.tokenizer_ckpt == r"runs\checkpoints\tokenizer\fsq_g8_v1024.pt"


@pytest.mark.parametrize("backbone", (Backbone.MAMBA, Backbone.TRANSFORMER))
def test_precision_case_reads_native_kernel_config(backbone, monkeypatch, tmp_path):
    config = probe.load_config("configs\\generator\\size_sweep\\width_320.yaml")
    tokens = torch.zeros(1, 4, 8, dtype=torch.long)
    trainer = SimpleNamespace(
        token_generator=SimpleNamespace(state_dict=lambda: {"weight": torch.ones(1)}, end_id=None),
        tokenizer_network=SimpleNamespace(encode=lambda motion: tokens),
        train_step=lambda *args: None,
    )
    monkeypatch.setattr(probe, "build_trainer", lambda *args: trainer)
    monkeypatch.setattr(probe, "seed_everything", lambda *args: None)
    monkeypatch.setattr(probe, "apply_precision", lambda *args: None)
    monkeypatch.setattr(
        probe, "fixed_batch", lambda *args: (torch.zeros(1), torch.zeros(1), torch.ones(1))
    )
    monkeypatch.setattr(
        probe,
        "precision_numeric",
        lambda *args: (np.ones(1), np.ones(1), 1.0, ["weight"]),
    )
    monkeypatch.setattr(probe, "median_ms", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1000)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = Namespace(
        batch_size=1,
        frames=[16],
        repeats=1,
        tokenizer_ckpt="unused",
        out_path=tmp_path / "case.json",
    )
    row, _, _, returned = probe.precision_case(args, config, "cpu", backbone, "fp32", None)
    assert row["mamba_kernel"] is (backbone == Backbone.MAMBA)
    torch.testing.assert_close(returned, tokens)


def test_step_timer_preserves_samples_without_warmup(monkeypatch):
    clock = iter((0.0, 0.001, 0.001, 0.003, 0.003, 0.006))
    calls, durations = [], []
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    measured = probe.median_ms(lambda: calls.append(None), 3, durations)
    assert len(calls) == 4
    assert durations == pytest.approx([1, 2, 3])
    assert measured == pytest.approx(2)
