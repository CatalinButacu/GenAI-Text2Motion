"""Strong RVQ baseline (Contribution A's baseline-to-beat): I/O contract matches the FSQ tokenizer,
the EMA codebook updates without collapsing, indices round-trip, and it overfits one batch (the
mandatory sanity gate before any real tokenizer run)."""

import torch

from text2motion.model.rvq_baseline import RvqBaselineTokenizer
from text2motion.model.tokenizer import reconstruction_loss
from text2motion.shared.config import RvqBaselineCfg

CFG = RvqBaselineCfg(
    in_dim=263,
    width=32,
    downsample=4,
    n_resblocks=1,
    num_quantizers=2,
    codebook_size=16,
    code_dim=8,
    quant_dropout=0.0,  # deterministic recon for the sanity gate
)


def test_encode_decode_shapes_and_contract():
    tok = RvqBaselineTokenizer(CFG).eval()
    x = torch.randn(2, 32, 263)

    indices = tok.encode(x)
    assert indices.shape == (2, 32 // CFG.downsample, CFG.num_quantizers)
    assert indices.dtype == torch.long and indices.max() < CFG.codebook_size

    recon = tok.decode(indices)
    assert recon.shape == (2, 32, 263)

    full_recon, full_idx, commit = tok(x)
    assert full_recon.shape == (2, 32, 263)
    assert commit.ndim == 0  # scalar commitment loss
    # eval-mode forward and decode(encode) take the same argmin path -> identical
    assert torch.allclose(full_recon, recon)
    assert torch.equal(full_idx, indices)


def test_overfit_one_batch():
    torch.manual_seed(0)
    tok = RvqBaselineTokenizer(CFG).train()
    opt = torch.optim.AdamW(tok.parameters(), lr=3e-3)
    x = torch.randn(2, 32, 263)

    def step() -> float:
        recon, _, commit = tok(x)
        loss = reconstruction_loss(recon, x) + CFG.commitment_beta * commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        return reconstruction_loss(recon, x).item()

    first = step()
    last = first
    for _ in range(200):
        last = step()

    assert last < 0.4 * first  # the recon path (enc/dec + EMA codebook) actually learns


def test_codebook_updates_and_no_collapse():
    torch.manual_seed(0)
    tok = RvqBaselineTokenizer(CFG).train()
    opt = torch.optim.AdamW(tok.parameters(), lr=3e-3)
    x = torch.randn(4, 64, 263)

    embed_before = tok.rvq.layers[0].embed.clone()
    for _ in range(50):
        recon, _, commit = tok(x)
        (reconstruction_loss(recon, x) + CFG.commitment_beta * commit).backward()
        opt.step()
        opt.zero_grad()

    # EMA moved the codebook (it is a buffer, not gradient-updated)
    assert not torch.allclose(embed_before, tok.rvq.layers[0].embed)
    # the first level uses more than one code (no single-code collapse)
    idx0 = tok.eval().encode(x)[..., 0]
    assert idx0.unique().numel() > 1
