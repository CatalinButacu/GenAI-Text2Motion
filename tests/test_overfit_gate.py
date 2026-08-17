from dataclasses import asdict, replace

import pytest
import torch

from text2motion.app.checkpoint import OverfitGate, sha256_file
from text2motion.app.config import Config
from text2motion.generation.model import GeneratorArchitecture


def resolve_generator_config(config: Config, backbone: str, codebook_size: int) -> dict:
    return GeneratorArchitecture.resolve(
        config.generator, backbone, codebook_size, config.tokenizer.num_quantizers
    ).as_dict()


def test_overfit_gate_is_bound_to_architecture_config_and_tokenizer(tmp_path):
    config_path = tmp_path / "config.yaml"
    tokenizer_path = tmp_path / "tokenizer.pt"
    gate_path = tmp_path / "gate.json"
    config_path.write_text("seed: 2026\n", encoding="utf-8")
    torch.save({"weight": torch.ones(1)}, tokenizer_path)
    cfg = Config()
    resolved = resolve_generator_config(cfg, "mamba", codebook_size=1024)
    gate = OverfitGate(
        config_sha256=sha256_file(config_path),
        tokenizer_sha256=sha256_file(tokenizer_path),
        backbone="mamba",
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
        backbone="mamba",
        resolved_config=resolved,
    )

    with pytest.raises(RuntimeError, match="backbone"):
        OverfitGate.load(gate_path).verify_setup(
            config_path=config_path,
            tokenizer_ckpt=tokenizer_path,
            backbone="transformer",
            resolved_config=asdict(replace(cfg.generator, backbone="transformer")),
        )


def test_overfit_gate_rejects_weak_accuracy(tmp_path):
    config_path = tmp_path / "config.yaml"
    tokenizer_path = tmp_path / "tokenizer.pt"
    gate_path = tmp_path / "gate.json"
    config_path.write_text("seed: 2026\n", encoding="utf-8")
    torch.save({}, tokenizer_path)
    resolved = resolve_generator_config(Config(), "mamba", codebook_size=1024)

    OverfitGate(
        config_sha256=sha256_file(config_path),
        tokenizer_sha256=sha256_file(tokenizer_path),
        backbone="mamba",
        resolved_generator_config=resolved,
        batch_size=4,
        steps=100,
        first_total=10.0,
        final_total=0.4,
        token_accuracy=0.98,
        ce=0.05,
    ).save(gate_path)

    with pytest.raises(RuntimeError, match="below 99%"):
        OverfitGate.load(gate_path).verify_setup(
            config_path=config_path,
            tokenizer_ckpt=tokenizer_path,
            backbone="mamba",
            resolved_config=resolved,
        )
