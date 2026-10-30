import pytest
import torch

from text2motion.app.config import TokenizerConfig
from text2motion.motion.model import MotionClip, MotionTokens
from text2motion.tokenization.nn_model import (
    FiniteScalarQuantizationTokenizer,
    FrozenMotionTokenizer,
    feature_velocity_reconstruction_loss,
)
from text2motion.tokenization.nn_quantizers import (
    FiniteScalarQuantizer,
    GroupedFiniteScalarQuantizer,
    typed_modules,
)

CFG = TokenizerConfig(
    in_dim=263, width=128, downsample=4, num_quantizers=4, fsq_levels=(8, 5, 5, 5)
)


def test_fsq_index_code_roundtrip():
    fsq = FiniteScalarQuantizer(CFG.fsq_levels)
    z = torch.randn(2, 16, len(CFG.fsq_levels))
    codes = fsq.quantize(z)
    idx = fsq.codes_to_indices(codes)

    assert idx.min() >= 0 and idx.max() < fsq.codebook_size
    assert torch.allclose(fsq.indices_to_codes(idx), codes, atol=1e-5)


def test_tokenizer_shapes_and_index_range():
    tok = FiniteScalarQuantizationTokenizer(CFG).eval()
    x = torch.randn(3, 64, 263)
    recon, idx, commit = tok(x)

    assert float(commit) == 0.0
    assert recon.shape == x.shape
    assert idx.shape == (3, 64 // CFG.downsample, CFG.num_quantizers)
    assert idx.dtype == torch.long
    assert idx.min() >= 0 and idx.max() < tok.codebook_size
    assert torch.allclose(tok.decode(tok.encode(x)), recon, atol=1e-5)


def test_single_batch_overfit():
    torch.manual_seed(0)
    tok = FiniteScalarQuantizationTokenizer(CFG).train()
    opt = torch.optim.Adam(tok.parameters(), lr=1e-3)
    x = torch.randn(2, 32, 263)

    recon0 = tok(x).reconstruction
    loss = feature_velocity_reconstruction_loss(recon0, x)
    loss0 = loss.item()

    for _ in range(200):
        recon = tok(x).reconstruction
        loss = feature_velocity_reconstruction_loss(recon, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < 0.5 * loss0  # must at least halve -- proves the path learns


def test_grouped_quantizer_matches_one_call_for_each_group():
    torch.manual_seed(0)
    quantizer = GroupedFiniteScalarQuantizer((8, 8, 4, 4), num_quantizers=3)
    latent = torch.randn(2, 5, 12) * 3
    codes, indices, _ = quantizer(latent)
    groups = typed_modules(quantizer.groups, FiniteScalarQuantizer)
    for group_index, (group, chunk) in enumerate(zip(groups, latent.split(4, dim=-1), strict=True)):
        group_codes = group.quantize(chunk)
        assert torch.equal(codes[..., 4 * group_index : 4 * group_index + 4], group_codes)
        assert torch.equal(indices[..., group_index], group.codes_to_indices(group_codes))
    assert torch.equal(quantizer.indices_to_codes(indices), codes)


def test_indices_with_another_codebook_count_raise():
    quantizer = GroupedFiniteScalarQuantizer((8, 8, 4, 4), num_quantizers=3)
    with pytest.raises(ValueError, match="4 codebooks"):
        quantizer.indices_to_codes(torch.zeros(2, 5, 4, dtype=torch.long))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_encode_of_non_finite_motion_raises(bad_value):
    tok = FiniteScalarQuantizationTokenizer(CFG).eval()
    motion = torch.randn(1, 16, 263)
    motion[0, 5, 7] = bad_value
    with pytest.raises(ValueError, match="NaN or Inf"):
        tok.encode(motion)


@pytest.mark.parametrize("frames", [0, 3, 62])
@pytest.mark.parametrize("call", ["encode", "forward"])
def test_a_frame_count_that_is_not_a_multiple_of_downsample_raises(frames, call):
    tok = FiniteScalarQuantizationTokenizer(CFG).eval()
    with pytest.raises(ValueError, match=f"has {frames} frames"):
        getattr(tok, call)(torch.randn(1, frames, 263))


def test_the_frozen_tokenizer_cuts_a_clip_to_whole_token_steps():
    frozen = FrozenMotionTokenizer(FiniteScalarQuantizationTokenizer(CFG))
    tokens = frozen.encode(MotionClip(features=torch.randn(62, 263), frame_count=62))
    assert tokens.frame_count == 60
    assert tokens.indices.shape == (15, CFG.num_quantizers)


def test_the_frozen_tokenizer_rejects_a_clip_shorter_than_one_token_step():
    frozen = FrozenMotionTokenizer(FiniteScalarQuantizationTokenizer(CFG))
    with pytest.raises(ValueError, match="fewer than one token"):
        frozen.encode(MotionClip(features=torch.randn(3, 263), frame_count=3))


def test_the_frozen_tokenizer_rejects_one_dimensional_indices():
    frozen = FrozenMotionTokenizer(FiniteScalarQuantizationTokenizer(CFG))
    tokens = MotionTokens(indices=torch.zeros(CFG.num_quantizers, dtype=torch.long), frame_count=4)
    with pytest.raises(ValueError, match="token indices must have shape"):
        frozen.decode(tokens)
