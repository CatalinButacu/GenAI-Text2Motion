from pathlib import Path

import pytest

from text2motion.app.cli import COMMANDS, parse_request
from text2motion.evaluation.contracts import BenchmarkRequest, StreamingBenchmarkProtocol


def test_benchmark_request_has_the_protocol_defaults():
    request = parse_request("benchmark", [])
    assert isinstance(request, BenchmarkRequest)
    assert request.horizons == StreamingBenchmarkProtocol().horizons
    assert request.config == Path("configs/default.yaml")
    assert request.report is None


def test_benchmark_request_reads_typed_values():
    request = parse_request(
        "benchmark", ["--horizons", "[4, 8]", "--device", "cpu", "--report", "out/bench.json"]
    )
    assert isinstance(request, BenchmarkRequest)
    assert request.horizons == (4, 8)
    assert request.device == "cpu"
    assert request.report == Path("out/bench.json")


def test_benchmark_request_rejects_a_wrong_type():
    with pytest.raises(SystemExit):
        parse_request("benchmark", ["--warmup", "many"])


def test_train_generator_has_no_option_to_select_on_the_test_split():
    required = [
        "--config",
        "configs/default.yaml",
        "--backbone",
        "mamba",
        "--overfit_gate",
        "g.json",
    ]
    parse_request("train-generator", required)

    with pytest.raises(SystemExit):
        parse_request("train-generator", [*required, "--eval_split", "test"])


def test_each_request_field_has_a_help_text():
    for command in COMMANDS.values():
        for name, field in command.request_model.model_fields.items():
            assert field.description, f"{command.request_model.__name__}.{name} has no help text"


def test_tokenize_request_needs_a_config():
    with pytest.raises(SystemExit):
        parse_request("tokenize", [])


def test_tokenize_request_reads_paths():
    request = parse_request("tokenize", ["--config", "c.yaml", "--token_pack", "data/t.npz"])
    assert request.model_dump()["config"] == Path("c.yaml")
    assert request.model_dump()["token_pack"] == Path("data/t.npz")


def test_health_request_defaults_to_the_config_address():
    request = parse_request("health", ["--port", "9000"])
    assert request.model_dump() == {
        "config": Path("configs/default.yaml"),
        "host": None,
        "port": 9000,
        "service_config": Path("configs/service.yml"),
    }


def test_enum_flags_read_the_lower_case_value():
    request = parse_request(
        "train-generator",
        [
            "--config",
            "c.yaml",
            "--backbone",
            "mamba",
            "--overfit_gate",
            "g.json",
        ],
    )
    dumped = request.model_dump()
    assert dumped["backbone"].name == "MAMBA"


def test_a_flag_without_a_value_is_true():
    arguments = ["--config", "c.yaml", "--backbone", "mamba", "--overfit_gate", "g.json"]
    assert parse_request("train-generator", arguments).model_dump()["resume"] is False
    with_flags = parse_request("train-generator", [*arguments, "--resume", "--fresh_selection"])
    assert with_flags.model_dump()["resume"] is True
    assert with_flags.model_dump()["fresh_selection"] is True


def test_every_command_parses_its_defaults_or_asks_for_required_flags():
    required = {
        "tokenize": ["--config", "c.yaml"],
        "train-tokenizer": ["--config", "c.yaml", "--tokenizer", "fsq"],
        "sanity-overfit": [
            "--config",
            "c.yaml",
            "--backbone",
            "mamba",
            "--tokenizer_ckpt",
            "t.pt",
            "--overfit_gate",
            "g.json",
        ],
        "train-generator": [
            "--config",
            "c.yaml",
            "--backbone",
            "mamba",
            "--overfit_gate",
            "g.json",
        ],
        "pretrain": ["--config", "c.yaml", "--backbone", "transformer"],
    }
    for command in COMMANDS:
        assert parse_request(command, required.get(command, [])) is not None
