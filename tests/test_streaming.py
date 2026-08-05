import queue as queue_mod

import torch

from text2motion.data.hml3d.joints import kinematic_bones, recover_skeleton
from text2motion.model.generator import MotionGenerator
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import GeneratorCfg, TokenizerCfg
from text2motion.stream.decode import (
    StreamingMotionDecoder,
    collect_stream,
    run_producer,
)

TOK = TokenizerCfg(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorCfg(
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
    tok = ResidualFsqTokenizer(TOK)
    mean = torch.zeros(263)
    std = torch.ones(263)
    return StreamingMotionDecoder(tok, mean, std, chunk_tokens=4)


def test_windowed_decode_covers_all_frames_and_is_deterministic():
    dec = make_decoder()
    tokens = [torch.randint(0, TOK.fsq_levels[0] ** 2, (1, TOK.num_quantizers)) for _ in range(10)]

    chunks = list(dec.stream_tokens(iter(tokens)))
    total = sum(c.shape[1] for c in chunks)

    assert total == 10 * TOK.downsample  # every token step yields downsample frames
    assert all(c.shape[2] == 263 for c in chunks)
    again = list(dec.stream_tokens(iter(tokens)))
    assert torch.allclose(torch.cat(chunks, 1), torch.cat(again, 1))


def test_end_to_end_stream_from_generator():
    dec = make_decoder()
    gen = MotionGenerator(GEN).eval()
    text = torch.randn(1, GEN.d_text)

    chunks = list(dec.stream(gen, text, num_steps=8, temperature=0.0))  # greedy -> deterministic
    motion = torch.cat(chunks, dim=1)

    assert motion.shape == (1, 8 * TOK.downsample, 263)


def test_queue_producer_pushes_chunks_then_sentinel():
    dec = make_decoder()
    gen = MotionGenerator(GEN).eval()
    q: queue_mod.Queue = queue_mod.Queue(maxsize=4)
    run_producer(dec, gen, torch.randn(1, GEN.d_text), num_steps=8, out_queue=q, temperature=0.0)

    motion = collect_stream(q)  # drains until STREAM_END
    assert motion.shape == (8 * TOK.downsample, 263)
    assert q.empty()


def test_producer_never_stalls_on_a_full_queue():
    dec = make_decoder()
    gen = MotionGenerator(GEN).eval()
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
