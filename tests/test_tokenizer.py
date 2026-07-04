"""Residual-FSQ tokenizer: shape contract, FSQ index<->code round-trip, and a single-batch
overfit (the sanity-overfit gate). No real data needed -- synthetic motion."""

import torch

from text2motion.model.tokenizer import FSQ, ResidualFsqTokenizer, reconstruction_loss
from text2motion.shared.config import TokenizerCfg

CFG = TokenizerCfg(in_dim=263, width=128, downsample=4, num_quantizers=4, fsq_levels=(8, 5, 5, 5))


def test_fsq_index_code_roundtrip():
    fsq = FSQ(CFG.fsq_levels)
    z = torch.randn(2, 16, len(CFG.fsq_levels))
    codes = fsq.quantize(z)
    idx = fsq.codes_to_indices(codes)

    assert idx.min() >= 0 and idx.max() < fsq.codebook_size
    # indices -> codes must invert codes_to_indices exactly
    assert torch.allclose(fsq.indices_to_codes(idx), codes, atol=1e-5)


def test_tokenizer_shapes_and_index_range():
    tok = ResidualFsqTokenizer(CFG).eval()
    x = torch.randn(3, 64, 263)
    recon, idx = tok(x)

    assert recon.shape == x.shape
    assert idx.shape == (3, 64 // CFG.downsample, CFG.num_quantizers)
    assert idx.dtype == torch.long
    assert idx.min() >= 0 and idx.max() < tok.codebook_size
    # encode/decode round-trip is shape-consistent and matches forward's recon
    assert torch.allclose(tok.decode(tok.encode(x)), recon, atol=1e-5)


def test_single_batch_overfit():
    """A tiny model must drive reconstruction loss down on one fixed batch (gradients flow)."""
    torch.manual_seed(0)
    tok = ResidualFsqTokenizer(CFG).train()
    opt = torch.optim.Adam(tok.parameters(), lr=1e-3)
    x = torch.randn(2, 32, 263)

    recon0, _ = tok(x)
    loss0 = reconstruction_loss(recon0, x).item()

    for _ in range(200):
        recon, _ = tok(x)
        loss = reconstruction_loss(recon, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < 0.5 * loss0  # must at least halve -- proves the path learns
