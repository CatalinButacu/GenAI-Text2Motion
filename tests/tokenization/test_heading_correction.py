import torch

from text2motion.tokenization.contracts import TokenizerConfig
from text2motion.tokenization.nn_tokenizer import (
    FiniteScalarQuantizationTokenizer,
    heading_reconstruction_loss,
)

FRAMES = 32
SPLIT = 16


def build(heading: bool, random_head: bool = False) -> FiniteScalarQuantizationTokenizer:
    config = TokenizerConfig(
        width=64,
        num_quantizers=2,
        n_resblocks=1,
        causal=True,
        heading_correction=heading,
    )
    torch.manual_seed(2026)
    module = FiniteScalarQuantizationTokenizer(config).eval()
    if random_head and module.heading_head is not None:
        torch.nn.init.normal_(module.heading_head.weight, std=0.1)
    return module


def test_a_zero_initialised_heading_head_leaves_the_output_unchanged():
    motion = torch.randn(2, FRAMES, 263, generator=torch.Generator().manual_seed(2026))
    with torch.no_grad():
        plain = build(heading=False)(motion)[0]
        corrected = build(heading=True)(motion)[0]
    assert torch.allclose(plain, corrected, atol=1e-6)


def test_the_heading_loss_is_zero_for_an_exact_reconstruction():
    motion = torch.randn(2, FRAMES, 263, generator=torch.Generator().manual_seed(2026))
    assert heading_reconstruction_loss(motion, motion).item() == 0.0


def test_the_heading_loss_grows_with_an_accumulating_turning_error():
    motion = torch.zeros(1, FRAMES, 263)
    reconstruction = motion.clone()
    reconstruction[..., 0] = 0.01
    assert heading_reconstruction_loss(reconstruction, motion).item() > 0.0


def test_a_causal_heading_correction_does_not_read_later_frames():
    generator = torch.Generator().manual_seed(2026)
    first = torch.randn(1, FRAMES, 263, generator=generator)
    second = first.clone()
    second[:, SPLIT:] = torch.randn(1, FRAMES - SPLIT, 263, generator=generator)
    module = build(heading=True, random_head=True)
    with torch.no_grad():
        before = module(first)[0][:, :SPLIT]
        after = module(second)[0][:, :SPLIT]
    assert torch.allclose(before, after, atol=1e-6)
