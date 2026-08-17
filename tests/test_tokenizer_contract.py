import pytest
import torch

from text2motion.app.config import RvqConfig, TokenizerConfig
from text2motion.tokenization.model import (
    DROPPED_CODE,
    MotionQuantizer,
    ResidualFsqTokenizer,
    RvqBaselineTokenizer,
    TokenizerModule,
    resample_stages,
)

FSQ_CFG = TokenizerConfig(
    in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(8, 8, 4, 4), n_resblocks=1
)
RVQ_CFG = RvqConfig(
    in_dim=263,
    width=64,
    downsample=4,
    num_quantizers=2,
    codebook_size=64,
    code_dim=8,
    n_resblocks=1,
)


def both_tokenizers():
    return [ResidualFsqTokenizer(FSQ_CFG), RvqBaselineTokenizer(RVQ_CFG)]


@pytest.mark.parametrize("tok", both_tokenizers())
def test_both_families_satisfy_one_contract(tok):
    assert isinstance(tok, TokenizerModule)
    assert isinstance(tok.quantizer, MotionQuantizer)

    x = torch.randn(2, 32, 263)
    out = tok.eval()(x)

    assert out.recon.shape == x.shape
    assert out.indices.dtype == torch.long
    assert out.commit.ndim == 0
    assert len(out) == 3


@pytest.mark.parametrize("tok", both_tokenizers())
def test_quantizer_exposes_units_and_combine(tok):
    quant = tok.quantizer
    assert len(quant.units) == 2
    codes = [torch.randn(2, 8, quant.units[0].dim if hasattr(quant.units[0], "dim") else 8)]
    assert quant.combine_codes(codes * 2).ndim == 3


def test_dropped_levels_use_a_sentinel_not_code_zero():
    cfg = TokenizerConfig(
        in_dim=263,
        width=64,
        downsample=4,
        num_quantizers=4,
        fsq_levels=(8, 8, 4, 4),
        n_resblocks=1,
        quantizer="residual",
        quant_dropout=1.0,
    )
    tok = ResidualFsqTokenizer(cfg).train()
    torch.manual_seed(0)

    saw_dropped = False
    for _ in range(20):
        indices = tok(torch.randn(2, 32, 263)).indices
        if bool((indices == DROPPED_CODE).any()):
            saw_dropped = True
            break

    assert saw_dropped, "quant_dropout=1.0 should drop levels in at least one of 20 batches"

    with pytest.raises(RuntimeError, match="quantizer dropout"):
        tok.encode(torch.randn(2, 32, 263))


def test_eval_mode_never_drops_levels():
    tok = ResidualFsqTokenizer(
        TokenizerConfig(
            in_dim=263,
            width=64,
            downsample=4,
            num_quantizers=4,
            fsq_levels=(8, 8, 4, 4),
            n_resblocks=1,
            quantizer="residual",
            quant_dropout=1.0,
        )
    ).eval()

    assert int(tok.encode(torch.randn(2, 32, 263)).min()) >= 0


def test_resample_stages_rejects_non_power_of_two():
    assert resample_stages(4) == 2
    assert resample_stages(8) == 3
    with pytest.raises(ValueError, match="power of two"):
        resample_stages(3)
