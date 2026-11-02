import math
from dataclasses import asdict, replace

import pytest
import torch

from text2motion.app.checkpoint import OverfitGate, sha256_file
from text2motion.app.config import ApplicationConfig
from text2motion.generation.contracts import Backbone, OverfitCriteria
from text2motion.generation.nn_model import GeneratorModelSpec


def resolve_generator_config(config: ApplicationConfig, backbone: str, codebook_size: int) -> dict:
    return GeneratorModelSpec.resolve(
        config.generator, backbone, codebook_size, config.tokenizer.num_quantizers
    ).as_dict()


def test_overfit_gate_is_bound_to_architecture_config_and_tokenizer(tmp_path):
    config_path = tmp_path / "config.yaml"
    tokenizer_path = tmp_path / "tokenizer.pt"
    gate_path = tmp_path / "gate.json"
    config_path.write_text("seed: 2026\n", encoding="utf-8")
    torch.save({"weight": torch.ones(1)}, tokenizer_path)
    cfg = ApplicationConfig()
    resolved = resolve_generator_config(cfg, "mamba", codebook_size=1024)
    gate = OverfitGate(
        config_sha256=sha256_file(config_path),
        tokenizer_sha256=sha256_file(tokenizer_path),
        backbone=Backbone.MAMBA,
        resolved_generator_config=resolved,
        batch_size=4,
        steps=100,
        first_total=10.0,
        final_total=0.4,
        token_accuracy=0.995,
        ce=0.05,
    )
    gate.save(gate_path)

    OverfitGate.load(gate_path).verify_setup(
        config_path=config_path,
        tokenizer_ckpt=tokenizer_path,
        backbone=Backbone.MAMBA,
        resolved_config=resolved,
    )

    with pytest.raises(RuntimeError, match="backbone"):
        OverfitGate.load(gate_path).verify_setup(
            config_path=config_path,
            tokenizer_ckpt=tokenizer_path,
            backbone="transformer",
            resolved_config=asdict(replace(cfg.generator, backbone="transformer")),
        )


def test_overfit_gate_ignores_dropout_difference_from_training_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    tokenizer_path = tmp_path / "tokenizer.pt"
    gate_path = tmp_path / "gate.json"
    config_path.write_text("seed: 2026\n", encoding="utf-8")
    torch.save({"weight": torch.ones(1)}, tokenizer_path)
    cfg = ApplicationConfig()
    gate_resolved = resolve_generator_config(
        replace(cfg, generator=replace(cfg.generator, dropout=0.0)), "mamba", codebook_size=1024
    )
    train_resolved = resolve_generator_config(cfg, "mamba", codebook_size=1024)
    gate = OverfitGate(
        config_sha256=sha256_file(config_path),
        tokenizer_sha256=sha256_file(tokenizer_path),
        backbone=Backbone.MAMBA,
        resolved_generator_config=gate_resolved,
        batch_size=4,
        steps=100,
        first_total=10.0,
        final_total=0.4,
        token_accuracy=0.995,
        ce=0.05,
    )
    gate.save(gate_path)

    OverfitGate.load(gate_path).verify_setup(
        config_path=config_path,
        tokenizer_ckpt=tokenizer_path,
        backbone=Backbone.MAMBA,
        resolved_config=train_resolved,
    )


def __gate(**overrides) -> OverfitGate:
    fields = {
        "config_sha256": "config",
        "tokenizer_sha256": "tokenizer",
        "backbone": Backbone.MAMBA,
        "resolved_generator_config": {},
        "batch_size": 4,
        "steps": 100,
        "first_total": 10.0,
        "final_total": 0.4,
        "token_accuracy": 0.995,
        "ce": 0.05,
    }
    return OverfitGate(**{**fields, **overrides})


def test_overfit_gate_accepts_valid_metrics():
    __gate().validate_metrics()


@pytest.mark.parametrize(
    ("criteria", "message"),
    [
        (OverfitCriteria(min_token_accuracy=0.999), "token accuracy 0.995 is below 0.999"),
        (OverfitCriteria(max_ce=0.01), "CE 0.05 is above 0.01"),
        (OverfitCriteria(max_total_ratio=0.01), "total loss ratio 0.04 is above 0.01"),
    ],
)
def test_overfit_gate_reports_custom_criteria(criteria, message):
    gate = __gate()
    assert not criteria.accepts(gate.token_accuracy, gate.ce, gate.total_ratio)
    with pytest.raises(RuntimeError) as error:
        gate.validate_metrics(criteria)
    assert str(error.value) == f"overfit gate {message}"


@pytest.mark.parametrize(
    ("token_accuracy", "ce", "total_ratio", "expected"),
    [
        (0.99, 0.1, 0.05, True),
        (0.98, 0.1, 0.05, False),
        (0.99, 0.11, 0.05, False),
        (0.99, 0.1, 0.06, False),
        (math.nan, 0.1, 0.05, False),
        (0.99, math.nan, 0.05, False),
        (0.99, 0.1, math.nan, False),
    ],
)
def test_overfit_criteria_accepts_preserves_comparisons(token_accuracy, ce, total_ratio, expected):
    assert OverfitCriteria().accepts(token_accuracy, ce, total_ratio) is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "first_total": math.nan,
            "final_total": math.nan,
            "token_accuracy": math.nan,
            "ce": math.nan,
        },
        {"batch_size": 0, "steps": 0},
        {"final_total": math.inf},
        {"token_accuracy": 1.5},
        {"ce": -0.1},
    ],
    ids=["all_nan", "zero_counts", "infinite_total", "accuracy_above_one", "negative_ce"],
)
def test_overfit_gate_rejects_invalid_metrics(overrides):
    with pytest.raises(RuntimeError):
        __gate(**overrides).validate_metrics()


def test_overfit_gate_rejects_weak_accuracy(tmp_path):
    config_path = tmp_path / "config.yaml"
    tokenizer_path = tmp_path / "tokenizer.pt"
    gate_path = tmp_path / "gate.json"
    config_path.write_text("seed: 2026\n", encoding="utf-8")
    torch.save({}, tokenizer_path)
    resolved = resolve_generator_config(ApplicationConfig(), "mamba", codebook_size=1024)

    OverfitGate(
        config_sha256=sha256_file(config_path),
        tokenizer_sha256=sha256_file(tokenizer_path),
        backbone=Backbone.MAMBA,
        resolved_generator_config=resolved,
        batch_size=4,
        steps=100,
        first_total=10.0,
        final_total=0.4,
        token_accuracy=0.98,
        ce=0.05,
    ).save(gate_path)

    with pytest.raises(RuntimeError, match=r"below 0\.99"):
        OverfitGate.load(gate_path).verify_setup(
            config_path=config_path,
            tokenizer_ckpt=tokenizer_path,
            backbone=Backbone.MAMBA,
            resolved_config=resolved,
        )
