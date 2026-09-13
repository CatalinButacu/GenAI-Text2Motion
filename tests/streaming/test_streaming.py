import queue as queue_mod

import numpy as np
import pytest
import torch

from text2motion.app.config import GeneratorConfig, TokenizerConfig
from text2motion.generation.model import MotionTokenGenerator
from text2motion.motion.representation import kinematic_bones, recover_skeleton
from text2motion.streaming.decoder import (
    StreamingMotionDecoder,
    collect_stream,
    run_producer,
)
from text2motion.tokenization.model import FsqTokenizer

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
    backbone="mamba",
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
    tok = FsqTokenizer(TOK)
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
        tok = FsqTokenizer(TOK)
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

    tok = FsqTokenizer(TOK)
    decoder = StreamingMotionDecoder(
        tok, 4, torch.zeros(263), torch.ones(263), left_context=1, lookahead=1
    )
    list(
        decoder.stream(FakeGenerator(), torch.zeros(1, 1, 512), 4, cfg_scale=6.0, stop_at_end=True)
    )

    assert seen["cfg_scale"] == 6.0
    assert seen["stop_at_end"] is True


def test_unbounded_stream_runs_past_the_position_table_on_a_recurrent_backbone():
    import itertools

    cfg = GeneratorConfig(
        backbone="mamba",
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
        backbone="transformer",
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

    budget = gen.backbone.max_positions - cfg.text_prefix_len
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
