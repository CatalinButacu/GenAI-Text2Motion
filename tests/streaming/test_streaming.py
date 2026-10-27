import queue as queue_mod
from typing import cast

import numpy as np
import pytest
import torch

from text2motion.app.config import GeneratorConfig, TokenizerConfig
from text2motion.generation.contracts import Backbone
from text2motion.generation.nn_model import MotionTokenGenerator
from text2motion.motion.representation import kinematic_bones, recover_skeleton
from text2motion.streaming.decoder import (
    StreamingMotionDecoder,
    collect_stream,
    run_producer,
)
from text2motion.tokenization.nn_model import FiniteScalarQuantizationTokenizer

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
    backbone=Backbone.MAMBA,
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
)


def make_decoder() -> StreamingMotionDecoder:
    tok = FiniteScalarQuantizationTokenizer(TOK)
    mean = torch.zeros(263)
    std = torch.ones(263)
    return StreamingMotionDecoder(tok, 4, mean, std, chunk_tokens=4)


def test_windowed_decode_covers_all_frames_and_is_deterministic():
    dec = make_decoder()
    tokens = [torch.randint(0, TOK.fsq_levels[0] ** 2, (1, TOK.num_quantizers)) for _ in range(10)]

    chunks = list(dec.stream_tokens(iter(tokens)))
    total = sum(c.shape[1] for c in chunks)

    assert total == 10 * TOK.downsample  # every token step yields downsample frames
    assert all(c.shape[2] == 263 for c in chunks)
    again = list(dec.stream_tokens(iter(tokens)))
    assert torch.allclose(torch.cat(chunks, 1), torch.cat(again, 1))


def test_streamed_decode_equals_whole_sequence_decode():
    dec = make_decoder()
    steps = 24
    tokens = [
        torch.randint(0, TOK.fsq_levels[0] ** 2, (1, TOK.num_quantizers)) for _ in range(steps)
    ]

    streamed = torch.cat(list(dec.stream_tokens(iter(tokens))), dim=1)
    whole = dec.decode_tokens(torch.stack(tokens, dim=1))

    assert streamed.shape == whole.shape
    assert torch.allclose(streamed, whole, atol=1e-4)


def test_decoder_context_is_measured_and_state_stays_bounded():
    dec = make_decoder()

    assert dec.left_context > 0 and dec.lookahead > 0  # padded convs look both ways
    assert dec.lookahead_frames == dec.lookahead * TOK.downsample

    long_stream = dec.state_tokens * 10
    tokens = [
        torch.randint(0, TOK.fsq_levels[0] ** 2, (1, TOK.num_quantizers))
        for _ in range(long_stream)
    ]
    chunks = list(dec.stream_tokens(iter(tokens)))

    assert sum(c.shape[1] for c in chunks) == long_stream * TOK.downsample  # horizon-independent


def test_end_to_end_stream_from_generator():
    dec = make_decoder()
    gen = MotionTokenGenerator(GEN).eval()
    text = torch.randn(1, GEN.d_text)

    chunks = list(dec.stream(gen, text, num_steps=8, temperature=0.0))  # greedy -> deterministic
    motion = torch.cat(chunks, dim=1)

    assert motion.shape == (1, 8 * TOK.downsample, 263)


def test_queue_producer_pushes_chunks_then_sentinel():
    dec = make_decoder()
    gen = MotionTokenGenerator(GEN).eval()
    q: queue_mod.Queue = queue_mod.Queue(maxsize=4)
    run_producer(dec, gen, torch.randn(1, GEN.d_text), num_steps=8, out_queue=q, temperature=0.0)

    motion = collect_stream(q)  # drains until STREAM_END
    assert motion.shape == (8 * TOK.downsample, 263)
    assert q.empty()


def test_producer_never_stalls_on_a_full_queue():
    dec = make_decoder()
    gen = MotionTokenGenerator(GEN).eval()
    q: queue_mod.Queue = queue_mod.Queue(maxsize=1)
    run_producer(dec, gen, torch.randn(1, GEN.d_text), num_steps=8, out_queue=q, temperature=0.0)

    motion = collect_stream(q)  # returns -> proves no deadlock; sentinel was reached
    assert motion.shape[1] == 263  # some (possibly dropped) frames, then STREAM_END
    assert q.empty()


def test_recover_skeleton_and_bones():
    joints = recover_skeleton(torch.randn(12, 263))
    assert joints.shape == (12, 22, 3)

    bones = kinematic_bones()
    assert bones.ndim == 2 and bones.shape[1] == 2
    assert bones.min() >= 0 and bones.max() < 22


def test_explicit_context_skips_the_probe():
    import text2motion.streaming.decoder as decode_mod

    calls = []
    original = decode_mod.measure_decoder_context

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    decode_mod.measure_decoder_context = counting
    try:
        tok = FiniteScalarQuantizationTokenizer(TOK)
        mean = torch.zeros(263)
        std = torch.ones(263)

        decode_mod.StreamingMotionDecoder(tok, 4, mean, std, left_context=4, lookahead=4)
        assert calls == [], "explicit context must not pay for the probe"

        decode_mod.StreamingMotionDecoder(tok, 4, mean, std)
        assert len(calls) == 1
    finally:
        decode_mod.measure_decoder_context = original


def test_stream_forwards_guidance_and_end_stopping():
    seen = {}

    class FakeGenerator:
        def stream_token_indices(self, text_emb, steps, **kwargs):
            seen.update(kwargs)
            for _ in range(steps):
                yield torch.zeros(1, 8, dtype=torch.long)

    tok = FiniteScalarQuantizationTokenizer(TOK)
    decoder = StreamingMotionDecoder(
        tok, 4, torch.zeros(263), torch.ones(263), left_context=1, lookahead=1
    )
    list(
        decoder.stream(
            cast(MotionTokenGenerator, FakeGenerator()),
            torch.zeros(1, 1, 512),
            4,
            cfg_scale=6.0,
            stop_at_end=True,
        )
    )

    assert seen["cfg_scale"] == 6.0
    assert seen["stop_at_end"] is True


def test_unbounded_stream_runs_past_the_position_table_on_a_recurrent_backbone():
    import itertools

    cfg = GeneratorConfig(
        backbone=Backbone.MAMBA,
        text_prefix_len=4,
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
    gen = MotionTokenGenerator(cfg).eval()
    assert gen.backbone.max_positions is None

    stream = gen.stream_token_indices(torch.randn(1, 4, cfg.d_text), 0)
    tokens = list(itertools.islice(stream, cfg.max_seq_len * 3))
    stream.close()

    assert len(tokens) == cfg.max_seq_len * 3
    assert all(t.shape == (1, cfg.num_codebooks) for t in tokens)


def test_positional_backbone_refuses_unbounded_and_overlong_streams():
    cfg = GeneratorConfig(
        backbone=Backbone.TRANSFORMER,
        text_prefix_len=4,
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
    gen = MotionTokenGenerator(cfg).eval()
    text = torch.randn(1, 4, cfg.d_text)
    assert gen.backbone.max_positions == cfg.max_seq_len + 1

    with pytest.raises(RuntimeError, match="cannot stream unboundedly"):
        list(gen.stream_token_indices(text, 0))

    with pytest.raises(ValueError, match="supports at most"):
        list(gen.stream_token_indices(text, cfg.max_seq_len * 4))

    positions = gen.backbone.max_positions
    assert positions is not None
    budget = positions - cfg.text_prefix_len
    assert len(list(gen.stream_token_indices(text, budget))) == budget


def test_streamed_root_recovery_matches_whole_sequence_trajectory():
    from text2motion.motion.representation import (
        StreamingSkeletonRecovery,
        recover_skeleton,
    )

    frames, chunk = 96, 16
    feat = np.zeros((frames, 263), dtype=np.float32)
    feat[:, 0] = 0.04
    feat[:, 1] = 0.05
    feat[:, 3] = 0.9

    whole = recover_skeleton(feat)
    recovery = StreamingSkeletonRecovery()
    streamed = np.concatenate(
        [recovery(feat[i : i + chunk]) for i in range(0, frames, chunk)], axis=0
    )

    assert streamed.shape == whole.shape
    assert np.abs(streamed - whole).max() < 1e-4

    straight = np.zeros((frames, 263), dtype=np.float32)
    straight[:, 1] = 0.05
    straight[:, 3] = 0.9

    naive = np.concatenate(
        [recover_skeleton(straight[i : i + chunk]) for i in range(0, frames, chunk)], axis=0
    )
    recovery.reset()
    fixed = np.concatenate(
        [recovery(straight[i : i + chunk]) for i in range(0, frames, chunk)], axis=0
    )
    travelled = np.linalg.norm(fixed[-1, 0, [0, 2]] - fixed[0, 0, [0, 2]])
    naive_travelled = np.linalg.norm(naive[-1, 0, [0, 2]] - naive[0, 0, [0, 2]])
    assert travelled > naive_travelled * 4


def test_streaming_recovery_resets_to_the_origin():
    from text2motion.motion.representation import StreamingSkeletonRecovery

    feat = np.zeros((16, 263), dtype=np.float32)
    feat[:, 1] = 0.05
    feat[:, 3] = 0.9

    recovery = StreamingSkeletonRecovery()
    first = recovery(feat)
    assert recovery.offset[0] != 0.0

    recovery.reset()
    assert recovery.yaw == 0.0
    again = recovery(feat)
    assert np.allclose(first, again)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda graphs need a cuda device")
def test_cuda_graph_decode_streams_the_same_tokens_as_eager_decode():
    from text2motion.generation.nn_model import seeded_sampling_uniforms

    torch.manual_seed(0)
    gen = MotionTokenGenerator(GEN).cuda().eval()
    text = torch.randn(1, 32, device="cuda")
    steps = 12
    uniforms = seeded_sampling_uniforms([3], steps, GEN.num_codebooks, "cuda")

    def stream() -> torch.Tensor:
        tokens = gen.stream_token_indices(text, steps, cfg_scale=3.0, uniforms=uniforms)
        return torch.stack(list(tokens), 1)

    eager = stream()
    gen.enable_cuda_graph_decode()
    first = stream()
    replayed = stream()
    assert torch.equal(eager, first)
    assert torch.equal(eager, replayed)


def test_cuda_graph_decode_refuses_a_positional_backbone():
    cfg = GeneratorConfig(
        backbone=Backbone.TRANSFORMER,
        d_model=64,
        n_layers=2,
        d_text=32,
        num_codebooks=2,
        codebook_size=16,
        max_seq_len=32,
        n_heads=4,
    )
    with pytest.raises(ValueError, match="recurrent state"):
        MotionTokenGenerator(cfg).enable_cuda_graph_decode()


def test_anchored_stream_runs_a_positional_backbone_past_its_learned_positions():
    cfg = GeneratorConfig(
        backbone=Backbone.TRANSFORMER,
        d_model=64,
        n_layers=2,
        d_text=32,
        num_codebooks=2,
        codebook_size=16,
        max_seq_len=24,
        n_heads=4,
    )
    torch.manual_seed(0)
    gen = MotionTokenGenerator(cfg).eval()
    text = torch.randn(1, 32)
    limit = gen.backbone.max_positions
    assert limit is not None
    tokens = list(gen.stream_token_indices(text, 3 * limit, anchor_every=8, anchor_context=8))
    assert len(tokens) == 3 * limit


def test_anchoring_that_would_overrun_the_learned_positions_is_refused():
    cfg = GeneratorConfig(
        backbone=Backbone.TRANSFORMER,
        d_model=64,
        n_layers=2,
        d_text=32,
        num_codebooks=2,
        codebook_size=16,
        max_seq_len=24,
        n_heads=4,
    )
    gen = MotionTokenGenerator(cfg).eval()
    with pytest.raises(ValueError, match="re-anchoring"):
        list(gen.stream_token_indices(torch.randn(1, 32), 10, anchor_every=20, anchor_context=16))


def test_anchoring_disabled_leaves_the_stream_unchanged_and_enabled_changes_it():
    from text2motion.generation.nn_model import seeded_sampling_uniforms

    torch.manual_seed(0)
    gen = MotionTokenGenerator(GEN).eval()
    text = torch.randn(1, 32)
    uniforms = seeded_sampling_uniforms([5], 20, GEN.num_codebooks, "cpu")
    plain = torch.stack(list(gen.stream_token_indices(text, 20, uniforms=uniforms)), 1)
    again = torch.stack(
        list(gen.stream_token_indices(text, 20, uniforms=uniforms, anchor_every=0)), 1
    )
    anchored = torch.stack(
        list(
            gen.stream_token_indices(text, 20, uniforms=uniforms, anchor_every=4, anchor_context=2)
        ),
        1,
    )
    assert torch.equal(plain, again)
    assert torch.equal(plain[:, :5], anchored[:, :5])
    assert not torch.equal(plain, anchored)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda graphs need a cuda device")
def test_anchored_cuda_graph_decode_matches_anchored_eager_decode():
    from text2motion.generation.nn_model import seeded_sampling_uniforms

    torch.manual_seed(0)
    gen = MotionTokenGenerator(GEN).cuda().eval()
    text = torch.randn(1, 32, device="cuda")
    uniforms = seeded_sampling_uniforms([9], 24, GEN.num_codebooks, "cuda")

    def stream() -> torch.Tensor:
        tokens = gen.stream_token_indices(
            text, 24, cfg_scale=3.0, uniforms=uniforms, anchor_every=6, anchor_context=4
        )
        return torch.stack(list(tokens), 1)

    eager = stream()
    gen.enable_cuda_graph_decode()
    assert torch.equal(eager, stream())


def test_a_new_text_at_a_reset_changes_only_the_tokens_after_that_reset():
    from text2motion.generation.nn_model import seeded_sampling_uniforms

    torch.manual_seed(0)
    gen = MotionTokenGenerator(GEN).eval()
    first_text = torch.randn(1, 32)
    second_text = torch.randn(1, 32)
    uniforms = seeded_sampling_uniforms([5], 12, GEN.num_codebooks, "cpu")
    settings = {"uniforms": uniforms, "anchor_every": 4, "anchor_context": 2, "cfg_scale": 3.0}
    one_text = torch.stack(list(gen.stream_token_indices(first_text, 12, **settings)), 1)
    switched = torch.stack(
        list(
            gen.stream_token_indices(
                first_text,
                12,
                text_at=lambda index: first_text if index < 8 else second_text,
                **settings,
            )
        ),
        1,
    )
    assert torch.equal(switched[:, :8], one_text[:, :8])
    assert not torch.equal(switched[:, 8:], one_text[:, 8:])


def test_a_text_schedule_without_resets_is_refused():
    gen = MotionTokenGenerator(GEN).eval()
    text = torch.randn(1, 32)
    with pytest.raises(ValueError, match="anchor_every"):
        list(gen.stream_token_indices(text, 4, text_at=lambda index: text))
