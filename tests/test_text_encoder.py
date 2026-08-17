import pytest
import torch

from text2motion.app.config import (
    GeneratorConfig,
    TextEncoderConfig,
    TokenizerConfig,
    TrainingConfig,
)
from text2motion.generation.model import MotionGeneratorModule
from text2motion.generation.text import CLIPTextEncoder
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.tokenization.model import ResidualFsqTokenizer

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
    backbone="mamba",
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


def _encoder(unfreeze_last_n: int = 1) -> CLIPTextEncoder:
    try:
        return CLIPTextEncoder(TextEncoderConfig(unfreeze_last_n=unfreeze_last_n))
    except Exception as exc:  # noqa: BLE001 - any load failure (offline / no weights) -> skip
        pytest.skip(f"CLIP weights unavailable: {exc}")


@pytest.mark.slow
def test_encode_shape_and_partial_unfreeze():
    enc = _encoder(unfreeze_last_n=1)
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
    enc = _encoder(unfreeze_last_n=1)
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGeneratorModule(GEN)
    trainer = GeneratorTrainer(
        gen, tok, TrainingConfig(cfg_dropout=0.0), downsample=4, text_encoder=enc
    )

    optimised = {id(p) for group in trainer.opt.param_groups for p in group["params"]}
    trainable_encoder = [p for p in enc.parameters() if p.requires_grad]
    assert trainable_encoder
    assert all(id(p) in optimised for p in trainable_encoder)

    encoder_lrs = {
        group["lr"]
        for group in trainer.opt.param_groups
        if any(id(p) in {id(q) for q in trainable_encoder} for p in group["params"])
    }
    assert encoder_lrs == {TrainingConfig().text_encoder_lr}

    text_emb = trainer.encode(["a person jumps", "a person sits down"])  # keeps grad to CLIP
    trainer.train_step(torch.randn(2, 32, 263), text_emb)

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters() if p.requires_grad
    )
