import torch

from text2motion.tokenization.contracts import TokenizerConfig
from text2motion.tokenization.nn_tokenizer import FiniteScalarQuantizationTokenizer

FRAMES = 32
SPLIT = 16
DOWNSAMPLE = 4


def build(causal: bool) -> FiniteScalarQuantizationTokenizer:
    config = TokenizerConfig(width=64, num_quantizers=2, n_resblocks=1, causal=causal)
    torch.manual_seed(2026)
    return FiniteScalarQuantizationTokenizer(config).eval()


def changed_future_frames() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2026)
    first = torch.randn(1, FRAMES, 263, generator=generator)
    second = first.clone()
    second[:, SPLIT:] = torch.randn(1, FRAMES - SPLIT, 263, generator=generator)
    return first, second


def encode(module: FiniteScalarQuantizationTokenizer, frames: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return module.encoder(frames)


def decode(module: FiniteScalarQuantizationTokenizer, latent: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return module.decoder(latent)


def test_a_causal_encoder_token_does_not_read_later_frames():
    first, second = changed_future_frames()
    module = build(causal=True)
    complete_tokens = SPLIT // DOWNSAMPLE
    before = encode(module, first)[:, :complete_tokens]
    after = encode(module, second)[:, :complete_tokens]
    assert torch.allclose(before, after, atol=1e-6)


def test_a_non_causal_encoder_token_reads_later_frames():
    first, second = changed_future_frames()
    module = build(causal=False)
    complete_tokens = SPLIT // DOWNSAMPLE
    before = encode(module, first)[:, :complete_tokens]
    after = encode(module, second)[:, :complete_tokens]
    assert not torch.allclose(before, after, atol=1e-6)


def test_a_causal_decoder_frame_does_not_read_later_tokens():
    generator = torch.Generator().manual_seed(2026)
    tokens = 8
    first = torch.randn(1, tokens, 64, generator=generator)
    second = first.clone()
    second[:, SPLIT // DOWNSAMPLE :] = torch.randn(
        1, tokens - SPLIT // DOWNSAMPLE, 64, generator=generator
    )
    module = build(causal=True)
    before = decode(module, first)[:, :SPLIT]
    after = decode(module, second)[:, :SPLIT]
    assert torch.allclose(before, after, atol=1e-6)


def test_a_non_causal_decoder_frame_reads_later_tokens():
    generator = torch.Generator().manual_seed(2026)
    tokens = 8
    first = torch.randn(1, tokens, 64, generator=generator)
    second = first.clone()
    second[:, SPLIT // DOWNSAMPLE :] = torch.randn(
        1, tokens - SPLIT // DOWNSAMPLE, 64, generator=generator
    )
    module = build(causal=False)
    before = decode(module, first)[:, :SPLIT]
    after = decode(module, second)[:, :SPLIT]
    assert not torch.allclose(before, after, atol=1e-6)


def test_the_causal_tokenizer_keeps_the_frame_count():
    motion = torch.randn(2, FRAMES, 263)
    module = build(causal=True)
    with torch.no_grad():
        reconstruction = module(motion)[0]
    assert reconstruction.shape == motion.shape
