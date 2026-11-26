from dataclasses import replace

import pytest
import torch

from text2motion.generation.contracts import Backbone, GeneratorConfig, StreamAnchors
from text2motion.generation.model.decode_step import EagerDecodeStep
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.sampling import seeded_sampling_uniforms
from text2motion.generation.model.transformer import TransformerBackbone


def small_cfg(backbone: Backbone) -> GeneratorConfig:
    return GeneratorConfig(
        backbone=backbone,
        d_model=64,
        n_layers=2,
        d_text=32,
        num_codebooks=2,
        codebook_size=16,
        max_seq_len=32,
        dropout=0.0,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank=8,
        n_heads=4,
        per_step_text=True,
    )


def trained_like(cfg: GeneratorConfig) -> MotionTokenGenerator:
    model = MotionTokenGenerator(cfg).eval()
    with torch.no_grad():
        model.text_proj.bias.normal_(std=0.5)
    return model


def decoded_logits(
    model: MotionTokenGenerator, tokens: torch.Tensor, text: torch.Tensor
) -> torch.Tensor:
    prefix = model.project_text_prefix(text)
    state = model.backbone.init_state(text.size(0), text.device)
    h, state = model.backbone.step(prefix[:, 0], state)
    step = EagerDecodeStep(model, state, model.per_step_text_vector(text, prefix))
    outputs = [model.predict_token_logits(h)]
    for feed in tokens[:, :-1].unbind(1):
        outputs.append(step(feed))
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("backbone", list(Backbone))
@torch.no_grad()
def test_a_null_text_row_gives_exactly_the_prefix_only_output(backbone):
    torch.manual_seed(2026)
    cfg = small_cfg(backbone)
    model = trained_like(cfg)
    prefix_only = MotionTokenGenerator(replace(cfg, per_step_text=False)).eval()
    prefix_only.load_state_dict(model.state_dict())
    tokens = torch.randint(cfg.codebook_size, (2, 8, cfg.num_codebooks))
    text = torch.randn(2, cfg.d_text)
    text[1].zero_()

    with_text = model(tokens, text)
    without_text = prefix_only(tokens, text)

    torch.testing.assert_close(with_text[1], without_text[1], rtol=0, atol=0)
    assert not torch.allclose(with_text[0], without_text[0])


@pytest.mark.parametrize("backbone", list(Backbone))
@torch.no_grad()
def test_eager_decode_with_per_step_text_matches_the_training_forward(backbone):
    torch.manual_seed(2026)
    cfg = small_cfg(backbone)
    model = trained_like(cfg)
    tokens = torch.randint(cfg.codebook_size, (2, 8, cfg.num_codebooks))
    text = torch.randn(2, cfg.d_text)
    text[1].zero_()

    torch.testing.assert_close(
        decoded_logits(model, tokens, text), model(tokens, text), atol=1e-4, rtol=1e-4
    )


@torch.no_grad()
def test_transformer_re_anchoring_over_the_full_history_keeps_the_per_step_text():
    torch.manual_seed(2026)
    cfg = small_cfg(Backbone.TRANSFORMER)
    model = trained_like(cfg)
    text = torch.randn(1, cfg.d_text)
    uniforms = seeded_sampling_uniforms([2026], 8, cfg.num_codebooks, "cpu")

    def streamed(anchor_every: int) -> torch.Tensor:
        return torch.stack(
            list(
                model.stream_token_indices(
                    text,
                    steps=8,
                    uniforms=uniforms,
                    anchors=StreamAnchors(every=anchor_every, context=16),
                )
            ),
            dim=1,
        )

    torch.testing.assert_close(streamed(3), streamed(0), rtol=0, atol=0)


@pytest.mark.parametrize("backbone", list(Backbone))
def test_default_reset_context_matches_explicit_constant_context(backbone):
    torch.manual_seed(2026)
    cfg = small_cfg(backbone)
    model = trained_like(cfg)
    text = torch.randn(1, cfg.d_text)
    uniforms = seeded_sampling_uniforms([2026], 25, cfg.num_codebooks, "cpu")
    original = torch.stack(
        list(
            model.stream_token_indices(
                text, 25, uniforms=uniforms, anchors=StreamAnchors(every=8, context=16)
            )
        ),
        dim=1,
    )
    explicit = torch.stack(
        list(
            model.stream_token_indices(
                text,
                25,
                uniforms=uniforms,
                anchors=StreamAnchors(every=8, context=16, context_at=lambda _: 16),
            )
        ),
        dim=1,
    )
    torch.testing.assert_close(original, explicit, rtol=0, atol=0)


@pytest.mark.parametrize(("keep", "expected"), [(0, [1, 9, 1, 9]), (8, [1, 9, 9, 17])])
def test_switch_discarded_history_does_not_resurrect_at_the_next_reset(monkeypatch, keep, expected):
    torch.manual_seed(2026)
    cfg = small_cfg(Backbone.TRANSFORMER)
    model = trained_like(cfg)
    calls = []
    transformer = model.backbone
    assert isinstance(transformer, TransformerBackbone)
    prefill = transformer.prefill

    def observed(sequence):
        calls.append(sequence.shape[1])
        return prefill(sequence)

    monkeypatch.setattr(transformer, "prefill", observed)
    list(
        model.stream_token_indices(
            torch.randn(1, cfg.d_text),
            25,
            anchors=StreamAnchors(
                every=8,
                context=16,
                context_at=lambda index: keep if index == 16 else 16,
            ),
        )
    )
    assert calls == expected


@pytest.mark.parametrize("keep", [-1, 17, 1.5, True])
def test_reset_context_rejects_invalid_or_unbounded_retention(keep):
    torch.manual_seed(2026)
    model = trained_like(small_cfg(Backbone.TRANSFORMER))
    with pytest.raises(ValueError, match="reset context"):
        list(
            model.stream_token_indices(
                torch.randn(1, 32), 10, anchors=StreamAnchors(every=8, context_at=lambda _: keep)
            )
        )


def test_reset_context_callback_requires_resets():
    model = trained_like(small_cfg(Backbone.TRANSFORMER))
    with pytest.raises(ValueError, match="require re-anchoring"):
        list(
            model.stream_token_indices(
                torch.zeros(1, 32), 1, anchors=StreamAnchors(context_at=lambda _: 0)
            )
        )


def test_per_step_text_refuses_a_multi_vector_prefix():
    with pytest.raises(ValueError, match="text_prefix_len must be 1"):
        MotionTokenGenerator(replace(small_cfg(Backbone.MAMBA), text_prefix_len=16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda graphs need a cuda device")
@torch.no_grad()
def test_graph_decode_follows_a_prompt_change_like_eager_decode():
    torch.manual_seed(2026)
    cfg = small_cfg(Backbone.MAMBA)
    model = trained_like(cfg).cuda()
    model.enable_cuda_graph_decode()
    tokens = torch.randint(cfg.codebook_size, (2, 6, cfg.num_codebooks), device="cuda")

    for _ in range(2):
        text = torch.randn(2, cfg.d_text, device="cuda")
        text[1].zero_()
        prefix = model.project_text_prefix(text)
        text_vector = model.per_step_text_vector(text, prefix)
        state = model.backbone.init_state(2, torch.device("cuda"))
        _, state = model.backbone.step(prefix[:, 0], state)
        eager = EagerDecodeStep(model, [tuple(t.clone() for t in s) for s in state], text_vector)
        captured = model.decode_step_for(state, 2, text_vector)
        for feed in tokens.unbind(1):
            expected = eager(feed)
            torch.testing.assert_close(captured(feed).clone(), expected, atol=1e-4, rtol=1e-4)
