"""CLIP text encoder: caption -> (B, 512) features, partial-unfreeze policy, and trainer wiring
(the encoder's unfrozen params join the optimiser and receive gradient). Marked `slow` because it
downloads CLIP ViT-B/32 weights; skips cleanly when they are unavailable (offline)."""

import pytest
import torch

from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import GeneratorCfg, TextEncoderCfg, TokenizerCfg, TrainCfg
from text2motion.train.trainer import GeneratorTrainer

TOK = TokenizerCfg(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorCfg(
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
        return CLIPTextEncoder(TextEncoderCfg(unfreeze_last_n=unfreeze_last_n))
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
    # only the last encoder layer (11), the final norm, and the projection are trainable
    assert all(
        ("encoder.layers.11" in n) or ("final_layer_norm" in n) or ("text_projection" in n)
        for n in trainable
    )
    assert any("embeddings" in n for n in frozen)  # token/pos embeddings stay frozen


@pytest.mark.slow
def test_trainer_includes_encoder_and_grad_flows():
    enc = _encoder(unfreeze_last_n=1)
    tok = ResidualFsqTokenizer(TOK)
    gen = MotionGenerator(GEN)
    trainer = GeneratorTrainer(gen, tok, TrainCfg(cfg_dropout=0.0), text_encoder=enc)

    # optimiser has two param groups: generator (lr) + unfrozen CLIP (text_encoder_lr)
    assert len(trainer.opt.param_groups) == 2
    assert trainer.opt.param_groups[1]["lr"] == TrainCfg().text_encoder_lr

    text_emb = trainer.encode(["a person jumps", "a person sits down"])  # keeps grad to CLIP
    trainer.train_step(torch.randn(2, 32, 263), text_emb)

    # at least one unfrozen CLIP param received a gradient through the shared graph
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters() if p.requires_grad
    )
