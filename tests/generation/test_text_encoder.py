from dataclasses import replace

import pytest
import torch

from text2motion.app.config import (
    GeneratorConfig,
    GeneratorTrainingConfig,
    TextEncoderConfig,
    TokenizerConfig,
)
from text2motion.generation.contracts import Backbone
from text2motion.generation.nn_model import CLIPTextEncoder, MotionTokenGenerator
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.tokenization.nn_model import FiniteScalarQuantizationTokenizer

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
    backbone=Backbone.MAMBA,
    d_model=64,
    n_layers=2,
    d_text=512,  # = CLIP projection dim
    num_codebooks=2,
    codebook_size=16,
    max_seq_len=16,
    d_state=8,
    d_conv=4,
    expand=2,
    dt_rank=8,
)


def encoder(unfreeze_last_n: int = 1) -> CLIPTextEncoder:
    try:
        return CLIPTextEncoder(TextEncoderConfig(unfreeze_last_n=unfreeze_last_n))
    except Exception as exc:  # noqa: BLE001 - any load failure (offline / no weights) -> skip
        pytest.skip(f"CLIP weights unavailable: {exc}")


def monolithic_forward(enc: CLIPTextEncoder, texts: list[str]) -> torch.Tensor:
    tokens = enc.tokens(texts)
    out = enc.model(**tokens)
    if enc.cfg.prefix_len == 1:
        return out.text_embeds
    hidden = out.last_hidden_state[:, : enc.cfg.prefix_len - 1]
    mask = tokens["attention_mask"][:, : enc.cfg.prefix_len - 1]
    hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
    return torch.cat([out.text_embeds.unsqueeze(1), hidden], dim=1)


def grads_of(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    grads = []
    for p in params:
        assert p.grad is not None
        grads.append(p.grad.clone())
    return grads


@pytest.mark.slow
@pytest.mark.parametrize("unfreeze_last_n", [0, 1, 2])
@pytest.mark.parametrize("prefix_len", [1, 3])
def test_split_forward_matches_monolithic(unfreeze_last_n: int, prefix_len: int):
    enc = encoder(unfreeze_last_n=unfreeze_last_n)
    enc.cfg = replace(enc.cfg, prefix_len=prefix_len)
    texts = [
        "a person jumps",
        "a person walks forward then stops",
        "someone waves with the right hand",
        "a person jumps over a low wall and lands softly on both feet",
    ]
    trainable = [p for p in enc.parameters() if p.requires_grad]

    expected = monolithic_forward(enc, texts)
    expected.sum().backward()
    expected_grads = grads_of(trainable)
    enc.zero_grad()

    actual = enc(texts)
    actual.sum().backward()

    assert torch.equal(actual, expected), f"max abs diff {(actual - expected).abs().max().item()}"
    for index, (got, want) in enumerate(zip(grads_of(trainable), expected_grads, strict=True)):
        assert torch.equal(got, want), (
            f"param {index} max abs grad diff {(got - want).abs().max().item()}"
        )


@pytest.mark.slow
def test_encode_shape_and_partial_unfreeze():
    enc = encoder(unfreeze_last_n=1)
    emb = enc(["a person walks forward then stops", "someone waves with the right hand"])

    assert emb.shape == (2, 512)

    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    frozen = {n for n, p in enc.named_parameters() if not p.requires_grad}
    assert trainable and frozen  # partially, not fully, frozen
    assert all(
        ("encoder.layers.11" in n) or ("final_layer_norm" in n) or ("text_projection" in n)
        for n in trainable
    )
    assert any("embeddings" in n for n in frozen)  # token/pos embeddings stay frozen


@pytest.mark.slow
def test_trainer_includes_encoder_and_grad_flows():
    enc = encoder(unfreeze_last_n=1)
    tok = FiniteScalarQuantizationTokenizer(TOK)
    gen = MotionTokenGenerator(GEN)
    trainer = GeneratorTrainer(
        gen, tok, GeneratorTrainingConfig(cfg_dropout=0.0), downsample=4, text_encoder=enc
    )

    optimised = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    trainable_encoder = [p for p in enc.parameters() if p.requires_grad]
    assert trainable_encoder
    assert all(id(p) in optimised for p in trainable_encoder)

    encoder_lrs = {
        group["lr"]
        for group in trainer.optimizer.param_groups
        if any(id(p) in {id(q) for q in trainable_encoder} for p in group["params"])
    }
    assert encoder_lrs == {GeneratorTrainingConfig().text_encoder_lr}

    text_emb = trainer.encode(["a person jumps", "a person sits down"])  # keeps grad to CLIP
    trainer.train_step(torch.randn(2, 32, 263), text_emb)

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters() if p.requires_grad
    )
