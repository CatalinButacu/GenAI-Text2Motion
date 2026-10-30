from dataclasses import asdict

import pytest
import torch

from text2motion.app.checkpoint import (
    GeneratorBundle,
    load_generator_checkpoint,
)
from text2motion.app.config import GeneratorConfig
from text2motion.generation.nn_model import MotionTokenGenerator


def test_generator_bundle_keeps_generator_encoder_and_tokenizer_identity(tmp_path):
    cfg = GeneratorConfig(
        d_model=32,
        n_layers=1,
        mamba_n_layers=1,
        d_text=16,
        num_codebooks=2,
        codebook_size=16,
        max_seq_len=8,
        d_state=4,
        dt_rank=4,
        n_heads=1,
    )
    generator = MotionTokenGenerator(cfg)
    encoder = torch.nn.Linear(5, 7)
    tokenizer = tmp_path / "tokenizer.pt"
    torch.save({"weight": torch.ones(1)}, tokenizer)
    path = tmp_path / "generator.pt"

    GeneratorBundle.capture(
        generator,
        encoder,
        backbone="mamba",
        tokenizer_ckpt=tokenizer,
        resolved_generator_config=asdict(cfg),
        epoch=3,
        validation_fid=1.25,
    ).save(path)

    gen_state, enc_state, bundle = load_generator_checkpoint(path)
    assert gen_state.keys() == generator.state_dict().keys()
    assert enc_state is not None
    assert enc_state.keys() == encoder.state_dict().keys()
    assert bundle is not None
    assert bundle.backbone == "mamba"
    bundle.verify_tokenizer(tokenizer)

    torch.save({"weight": torch.zeros(1)}, tokenizer)
    with pytest.raises(ValueError, match="does not match"):
        bundle.verify_tokenizer(tokenizer)


def test_legacy_generator_state_dict_loads_without_fabricating_encoder(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"layer.weight": torch.ones(2, 2)}, path)

    gen_state, enc_state, bundle = load_generator_checkpoint(path)

    assert "layer.weight" in gen_state
    assert enc_state is None
    assert bundle is None


@pytest.mark.parametrize(
    ("change", "raises"),
    [({"use_kernel": True}, False), ({"d_model": 64}, True), ({"use_end_token": True}, True)],
    ids=["execution_field", "model_size", "end_token"],
)
def test_a_bundle_refuses_a_config_that_builds_another_model(tmp_path, change, raises):
    cfg = GeneratorConfig(d_model=32, n_layers=1, num_codebooks=2, codebook_size=16, n_heads=1)
    tokenizer = tmp_path / "tokenizer.pt"
    torch.save({"weight": torch.ones(1)}, tokenizer)
    bundle = GeneratorBundle.capture(
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
        backbone="transformer",
        tokenizer_ckpt=tokenizer,
        resolved_generator_config=asdict(cfg),
        epoch=1,
        validation_fid=1.0,
    )
    other = asdict(cfg) | change

    if not raises:
        bundle.verify_generator_config(other)
        return
    with pytest.raises(ValueError, match="another generator config"):
        bundle.verify_generator_config(other)


@pytest.mark.parametrize(("per_step_text", "raises"), [(False, False), (True, True)])
def test_a_bundle_from_before_per_step_text_matches_only_the_prefix_only_config(
    tmp_path, per_step_text, raises
):
    cfg = GeneratorConfig(d_model=32, n_layers=1, num_codebooks=2, codebook_size=16, n_heads=1)
    tokenizer = tmp_path / "tokenizer.pt"
    torch.save({"weight": torch.ones(1)}, tokenizer)
    old_config = {key: value for key, value in asdict(cfg).items() if key != "per_step_text"}
    bundle = GeneratorBundle.capture(
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
        backbone="transformer",
        tokenizer_ckpt=tokenizer,
        resolved_generator_config=old_config,
        epoch=1,
        validation_fid=1.0,
    )
    current = asdict(cfg) | {"per_step_text": per_step_text}

    if not raises:
        bundle.verify_generator_config(current)
        return
    with pytest.raises(ValueError, match="per_step_text"):
        bundle.verify_generator_config(current)
