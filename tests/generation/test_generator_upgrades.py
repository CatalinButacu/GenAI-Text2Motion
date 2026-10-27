import importlib.util
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from text2motion.app.config import GeneratorConfig, GeneratorTrainingConfig, TokenizerConfig
from text2motion.generation.contracts import Backbone, SamplingConfig
from text2motion.generation.nn_model import MotionTokenGenerator
from text2motion.generation.nn_text import CLIPTextEncoder
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.tokenization.nn_model import (
    FiniteScalarQuantizationTokenizer,
    FrozenMotionTokenizer,
)

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
BASE = GeneratorConfig(
    d_model=64,
    n_layers=2,
    d_text=32,
    num_codebooks=2,
    codebook_size=16,
    max_seq_len=32,
    d_state=8,
    d_conv=4,
    expand=2,
    dt_rank=8,
    n_heads=4,
)

HAS_MAMBA_SSM = importlib.util.find_spec("mamba_ssm") is not None


@pytest.mark.parametrize("backbone", ["transformer", "mamba"])
def test_multi_token_prefix_forward_and_stream(backbone):
    cfg = replace(BASE, backbone=Backbone(backbone), text_prefix_len=4)
    gen = MotionTokenGenerator(cfg).eval()
    tokens = torch.randint(0, cfg.codebook_size, (3, 8, cfg.num_codebooks))
    text = torch.randn(3, 4, cfg.d_text)

    logits = gen(tokens, text)
    assert logits.shape == (3, 8, cfg.num_codebooks, cfg.codebook_size)

    streamed = torch.stack(list(gen.stream_token_indices(text, 8, cfg_scale=3.0)), dim=1)
    assert streamed.shape == (3, 8, cfg.num_codebooks)


def test_prefix_length_mismatch_fails_loud():
    cfg = replace(BASE, backbone=Backbone.TRANSFORMER, text_prefix_len=4)
    gen = MotionTokenGenerator(cfg)

    with pytest.raises(ValueError, match="text_prefix_len"):
        gen(torch.randint(0, 16, (2, 8, 2)), torch.randn(2, 3, cfg.d_text))


def test_prefix_parity_with_legacy_single_token():
    cfg = replace(BASE, backbone=Backbone.MAMBA)
    gen = MotionTokenGenerator(cfg).eval()
    tokens = torch.randint(0, cfg.codebook_size, (2, 8, cfg.num_codebooks))
    text = torch.randn(2, cfg.d_text)

    with torch.no_grad():
        from_2d = gen(tokens, text)
        from_3d = gen(tokens, text.unsqueeze(1))
    assert torch.allclose(from_2d, from_3d)


def test_end_token_targets_and_masked_stream():
    cfg = replace(BASE, backbone=Backbone.TRANSFORMER, use_end_token=True)
    gen = MotionTokenGenerator(cfg).eval()
    assert gen.end_id == cfg.codebook_size

    tok = FiniteScalarQuantizationTokenizer(TOK)
    trainer = GeneratorTrainer(
        gen, tok, GeneratorTrainingConfig(cfg_dropout=0.0, pkeep=1.0), downsample=4
    )
    tokens = torch.randint(0, cfg.codebook_size, (2, 6, cfg.num_codebooks))
    lengths = torch.tensor([12, 24])  # 3 and 6 tokens at downsample 4
    targets, token_lengths = trainer.append_end_targets(tokens, lengths)

    assert targets.shape == (2, 7, cfg.num_codebooks)
    assert (targets[0, 3] == gen.end_id).all() and (targets[1, 6] == gen.end_id).all()
    assert token_lengths.tolist() == [4, 7]

    parts = trainer.train_step(torch.randn(2, 24, 263), torch.randn(2, cfg.d_text), lengths)
    assert all(v == v for v in parts.values())  # finite

    text = torch.randn(1, cfg.d_text)
    fixed = torch.stack(
        list(gen.stream_token_indices(text, 10)), dim=1
    )  # fixed-length: END unreachable
    assert fixed.shape == (1, 10, cfg.num_codebooks) and (fixed != gen.end_id).all()

    stopped = list(
        gen.stream_token_indices(text, 10, stop_at_end=True)
    )  # END allowed but never yielded (B=1)
    assert len(stopped) <= 10
    assert all((step != gen.end_id).all() for step in stopped)


def test_end_cannot_end_a_stream_at_the_first_step(monkeypatch):
    cfg = replace(BASE, backbone=Backbone.TRANSFORMER, use_end_token=True)
    gen = MotionTokenGenerator(cfg).eval()
    end_id = gen.end_id
    assert end_id is not None

    def end_is_most_likely(h):
        logits = torch.zeros(h.shape[0], cfg.num_codebooks, cfg.codebook_size + 1)
        logits[..., end_id] = 50.0
        return logits

    monkeypatch.setattr(gen, "predict_token_logits", end_is_most_likely)
    steps = list(gen.stream_token_indices(torch.randn(1, cfg.d_text), 10, stop_at_end=True))

    assert len(steps) == 1
    assert (steps[0] != end_id).all()


def test_each_row_stops_at_its_own_end_and_the_decoder_never_sees_end():
    end_id = 16
    script = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[end_id, 9], [1, 2]],
        [[0, 0], [end_id, 5]],
    ]
    decoded_inputs = []

    def decode(indices):
        decoded_inputs.append(indices)
        return torch.zeros(indices.shape[0], indices.shape[1] * 4, 263)

    token_generator = SimpleNamespace(
        end_id=end_id,
        cfg=SimpleNamespace(num_codebooks=2),
        stream_token_indices=lambda *args, **kwargs: iter(torch.tensor(step) for step in script),
    )
    tokenizer = SimpleNamespace(downsample=4, network=SimpleNamespace(decode=decode))
    pipeline = TextToMotionGenerator(
        cast(MotionTokenGenerator, token_generator),
        cast(CLIPTextEncoder, lambda captions: torch.zeros(len(captions), 32)),
        cast(FrozenMotionTokenizer, tokenizer),
    )

    motions = pipeline.generate_batch(["a", "b"], 4, SamplingConfig(stop_at_end=True))

    assert motions is not None
    assert [motion.tokens.token_count for motion in motions] == [2, 3]
    assert [motion.motion.frame_count for motion in motions] == [8, 12]
    assert all(int(indices.max()) < end_id for indices in decoded_inputs)


@pytest.mark.skipif(
    HAS_MAMBA_SSM, reason="mamba-ssm installed: the loud-failure path can't trigger"
)
def test_use_kernel_without_mamba_ssm_fails_loud():
    cfg = replace(BASE, backbone=Backbone.MAMBA, use_kernel=True)
    gen = MotionTokenGenerator(cfg)

    with pytest.raises(ImportError):
        gen(torch.randint(0, 16, (2, 8, 2)), torch.randn(2, cfg.d_text))


@pytest.mark.skipif(not HAS_MAMBA_SSM, reason="mamba-ssm not installed (Linux/CUDA only)")
def test_kernel_matches_eager_scan():
    cfg_eager = replace(BASE, backbone=Backbone.MAMBA)
    cfg_kernel = replace(BASE, backbone=Backbone.MAMBA, use_kernel=True)
    torch.manual_seed(0)
    eager = MotionTokenGenerator(cfg_eager).cuda().eval()
    kernel = MotionTokenGenerator(cfg_kernel).cuda().eval()
    kernel.load_state_dict(eager.state_dict())

    tokens = torch.randint(0, cfg_eager.codebook_size, (2, 8, 2), device="cuda")
    text = torch.randn(2, cfg_eager.d_text, device="cuda")
    with torch.no_grad():
        assert torch.allclose(eager(tokens, text), kernel(tokens, text), atol=1e-4)


def test_config_rejects_prefix_len_mismatch():
    import dataclasses

    import pytest

    from text2motion.app.config import (
        ApplicationConfig,
        GeneratorConfig,
        TextEncoderConfig,
        validate_config,
    )

    good = ApplicationConfig(
        text_encoder=TextEncoderConfig(prefix_len=16),
        generator=GeneratorConfig(text_prefix_len=16, max_seq_len=96),
    )
    validate_config(good)

    bad = dataclasses.replace(
        good, generator=dataclasses.replace(good.generator, text_prefix_len=1)
    )
    with pytest.raises(ValueError, match="text_prefix_len"):
        validate_config(bad)


def test_config_rejects_position_budget_overflow():
    import dataclasses

    import pytest

    from text2motion.app.config import ApplicationConfig, GeneratorConfig, validate_config

    cfg = ApplicationConfig(generator=GeneratorConfig(text_prefix_len=1, max_seq_len=8))
    with pytest.raises(ValueError, match="max_seq_len"):
        validate_config(cfg)

    ok = dataclasses.replace(cfg, generator=dataclasses.replace(cfg.generator, max_seq_len=96))
    validate_config(ok)


def test_config_rejects_rvq_num_quantizers_mismatch():
    import dataclasses

    import pytest

    from text2motion.app.config import ApplicationConfig, validate_config
    from text2motion.tokenization.contracts import RvqConfig, TokenizerConfig, TokenizerKind

    good = ApplicationConfig(
        tokenizer=TokenizerConfig(kind=TokenizerKind.RVQ, num_quantizers=8),
        rvq_baseline=RvqConfig(num_quantizers=8),
    )
    validate_config(good)

    bad = dataclasses.replace(
        good, rvq_baseline=dataclasses.replace(good.rvq_baseline, num_quantizers=6)
    )
    with pytest.raises(ValueError, match="num_quantizers"):
        validate_config(bad)

    fsq = ApplicationConfig(
        tokenizer=TokenizerConfig(kind=TokenizerKind.FSQ, num_quantizers=8),
        rvq_baseline=RvqConfig(num_quantizers=6),
    )
    validate_config(fsq)
