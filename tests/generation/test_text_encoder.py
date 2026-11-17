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
from text2motion.generation.nn_model import (
    CLIPTextEncoder,
    FrozenTextStates,
    MotionTokenGenerator,
)
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.tokenization.nn_model import FiniteScalarQuantizationTokenizer

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
CACHE_CAPTIONS = (
    "a person jumps",
    "a person walks forward then stops",
    "someone waves with the right hand",
    "a person jumps over a low wall and lands softly on both feet",
    "a man kicks a ball",
    "a woman spins once and bows",
    "a person climbs up a ladder slowly",
    "someone crouches down to pick something up from the floor",
    "a person runs in a small circle",
    "a man sits on a chair and crosses his legs",
    "a person stretches both arms over the head",
    "someone walks backwards three steps",
    "a dancer raises one leg and turns",
    "a person throws a punch and steps back",
    "a child hops on one foot",
    "a person bends forward to touch the toes and stands up again",
    "someone drinks from a cup",
    "a person waves goodbye",
    "a soldier marches in a straight line and salutes",
    "a person walks up stairs while holding the rail",
)
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
def test_frozen_cache_matches_uncached_forward_and_grads(tmp_path):
    enc = encoder(unfreeze_last_n=1)
    cache = FrozenTextStates.build(enc, list(CACHE_CAPTIONS), 8, tmp_path / "states.npy")
    ordered = sorted(CACHE_CAPTIONS)
    trainable = [p for p in enc.parameters() if p.requires_grad]

    for texts in (ordered[:8], ordered[8:16]):
        enc.frozen_cache = None
        expected = enc(texts)
        expected.sum().backward()
        expected_grads = grads_of(trainable)
        enc.zero_grad()

        enc.frozen_cache = cache
        actual = enc(texts)
        actual.sum().backward()

        assert torch.equal(actual, expected), (
            f"max abs diff {(actual - expected).abs().max().item()}"
        )
        for index, (got, want) in enumerate(zip(grads_of(trainable), expected_grads, strict=True)):
            assert torch.equal(got, want), (
                f"param {index} max abs grad diff {(got - want).abs().max().item()}"
            )
        enc.zero_grad()

    assert cache.hits == 2
    assert cache.misses == 0


@pytest.mark.slow
def test_frozen_cache_miss_for_uncached_caption(tmp_path):
    enc = encoder(unfreeze_last_n=1)
    cache = FrozenTextStates.build(enc, list(CACHE_CAPTIONS), 8, tmp_path / "states.npy")
    texts = [*sorted(CACHE_CAPTIONS)[:7], "a person does a cartwheel"]

    with torch.no_grad():
        enc.frozen_cache = None
        expected = enc(texts)
        enc.frozen_cache = cache
        actual = enc(texts)

    assert torch.equal(actual, expected), f"max abs diff {(actual - expected).abs().max().item()}"
    assert cache.misses == 1
    assert cache.hits == 0


@pytest.mark.slow
def test_frozen_cache_load_rejects_changed_weights(tmp_path):
    enc = encoder(unfreeze_last_n=1)
    path = tmp_path / "states.npy"
    FrozenTextStates.build(enc, list(CACHE_CAPTIONS), 8, path)

    first_layer = enc.model.text_model.encoder.layers[0]
    with torch.no_grad():
        next(first_layer.parameters()).view(-1)[0].add_(1.0)

    with pytest.raises(RuntimeError, match="delete"):
        FrozenTextStates.load(enc, path)


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
