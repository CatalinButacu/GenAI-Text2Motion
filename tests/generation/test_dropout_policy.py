from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from text2motion.generation.contracts import Backbone, GeneratorConfig
from text2motion.generation.nn_model import MotionTokenGenerator


@pytest.mark.parametrize("backbone", list(Backbone))
@pytest.mark.parametrize("n_layers", [1, 3])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dropout", [0.1, 0.5])
def test_dropout_is_only_on_the_shared_embedding_input(
    backbone: Backbone, n_layers: int, training: bool, dropout: float
) -> None:
    torch.manual_seed(2026)
    cfg = GeneratorConfig(
        backbone=backbone,
        d_model=16,
        n_layers=n_layers,
        d_text=8,
        num_codebooks=2,
        codebook_size=16,
        max_seq_len=16,
        dropout=dropout,
        text_prefix_len=3,
        d_state=4,
        d_conv=4,
        expand=2,
        dt_rank=4,
        n_heads=4,
        use_kernel=False,
    )
    model = MotionTokenGenerator(cfg).train(training)
    tokens = torch.randint(cfg.codebook_size, (2, 8, cfg.num_codebooks))
    text = torch.randn(2, cfg.text_prefix_len, cfg.d_text)
    expected = torch.cat(
        [model.project_text_prefix(text), model.embed_motion_tokens(tokens)[:, :-1]], dim=1
    )
    sites = {
        name: module for name, module in model.named_modules() if isinstance(module, nn.Dropout)
    }
    assert list(sites) == ["emb_drop"]
    assert sites["emb_drop"].p == cfg.dropout
    assert sites["emb_drop"].training is training

    with (
        patch.object(F, "dropout", wraps=F.dropout) as apply_dropout,
        patch.object(
            F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
        ) as attention,
        patch.object(model.backbone, "forward", wraps=model.backbone.forward) as forward,
    ):
        rng_before = torch.random.get_rng_state().clone()
        logits = model(tokens, text)
        rng_after = torch.random.get_rng_state()

    assert logits.shape == (2, 8, cfg.num_codebooks, cfg.codebook_size)
    assert apply_dropout.call_count == 1
    call = apply_dropout.call_args
    assert call is not None
    torch.testing.assert_close(call.args[0], expected, rtol=0, atol=0)
    assert call.args[1] == cfg.dropout
    assert call.args[2] is training
    assert forward.call_count == 1
    backbone_call = forward.call_args
    assert backbone_call is not None
    received = backbone_call.args[0]

    if training:
        kept = received.ne(0)
        assert kept.any()
        assert (received.eq(0) & expected.ne(0)).any()
        torch.testing.assert_close(received[kept], expected[kept] / (1 - cfg.dropout))
        assert not torch.equal(rng_before, rng_after)
    else:
        torch.testing.assert_close(received, expected, rtol=0, atol=0)
        assert torch.equal(rng_before, rng_after)

    expected_attention_calls = n_layers if backbone is Backbone.TRANSFORMER else 0
    assert attention.call_count == expected_attention_calls
    for call in attention.call_args_list:
        assert call.kwargs.get("dropout_p", 0.0) == 0.0
        assert call.kwargs["is_causal"] is True
    if not training:
        torch.testing.assert_close(model(tokens, text), logits, rtol=0, atol=0)
