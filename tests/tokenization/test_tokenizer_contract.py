from dataclasses import replace

import pytest
import torch

from text2motion.app.config import ResidualVectorQuantizationConfig, TokenizerConfig
from text2motion.tokenization.contracts import DROPPED_CODE, FiniteScalarStackLayout, TokenizerKind
from text2motion.tokenization.nn_model import (
    FiniteScalarQuantizationTokenizer,
    ResidualVectorQuantizationTokenizer,
    TokenizerNetwork,
    build_tokenizer_network,
    downsample_stage_count,
)
from text2motion.tokenization.nn_quantizers import (
    FiniteScalarQuantizer,
    GroupedFiniteScalarQuantizer,
    QuantizerStack,
    ResidualFiniteScalarQuantizer,
    ResidualVectorQuantizer,
)

FSQ_CFG = TokenizerConfig(
    in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(8, 8, 4, 4), n_resblocks=1
)
RVQ_CFG = ResidualVectorQuantizationConfig(
    in_dim=263,
    width=64,
    downsample=4,
    num_quantizers=2,
    codebook_size=64,
    code_dim=8,
    n_resblocks=1,
)


def both_tokenizers():
    return [
        FiniteScalarQuantizationTokenizer(FSQ_CFG),
        ResidualVectorQuantizationTokenizer(RVQ_CFG),
    ]


@pytest.mark.parametrize("tok", both_tokenizers())
def test_both_families_satisfy_one_contract(tok):
    assert isinstance(tok, TokenizerNetwork)
    assert isinstance(tok.quantizer, QuantizerStack)

    x = torch.randn(2, 32, 263)
    out = tok.eval()(x)

    assert out.reconstruction.shape == x.shape
    assert out.token_indices.dtype == torch.long
    assert out.commitment_loss.ndim == 0
    assert len(out) == 3


@pytest.mark.parametrize("tok", both_tokenizers())
def test_quantizer_exposes_units_and_combine(tok):
    quant = tok.quantizer
    assert len(quant.quantizers) == 2
    unit = quant.quantizers[0]
    width = unit.dimension_count if hasattr(unit, "dimension_count") else 8
    codes = [torch.randn(2, 8, width)]
    assert quant.combine_code_vectors(codes * 2).ndim == 3


def test_dropped_levels_use_a_sentinel_not_code_zero():
    cfg = TokenizerConfig(
        in_dim=263,
        width=64,
        downsample=4,
        num_quantizers=4,
        fsq_levels=(8, 8, 4, 4),
        n_resblocks=1,
        quantizer=FiniteScalarStackLayout.RESIDUAL,
        quant_dropout=1.0,
    )
    tok = FiniteScalarQuantizationTokenizer(cfg).train()
    torch.manual_seed(0)

    saw_dropped = False
    for _ in range(20):
        indices = tok(torch.randn(2, 32, 263)).token_indices
        if bool((indices == DROPPED_CODE).any()):
            saw_dropped = True
            break

    assert saw_dropped, "quant_dropout=1.0 should drop levels in at least one of 20 batches"

    with pytest.raises(RuntimeError, match="quantizer dropout"):
        tok.encode(torch.randn(2, 32, 263))


def test_eval_mode_never_drops_levels():
    tok = FiniteScalarQuantizationTokenizer(
        TokenizerConfig(
            in_dim=263,
            width=64,
            downsample=4,
            num_quantizers=4,
            fsq_levels=(8, 8, 4, 4),
            n_resblocks=1,
            quantizer=FiniteScalarStackLayout.RESIDUAL,
            quant_dropout=1.0,
        )
    ).eval()

    assert int(tok.encode(torch.randn(2, 32, 263)).min()) >= 0


def test_resample_stages_rejects_non_power_of_two():
    assert downsample_stage_count(4) == 2
    assert downsample_stage_count(8) == 3
    with pytest.raises(ValueError, match="power of two"):
        downsample_stage_count(3)


@pytest.mark.parametrize("levels", [(2, 4), (1, 4), ()], ids=["level_two", "level_one", "empty"])
def test_fsq_levels_below_three_or_empty_raise(levels):
    with pytest.raises(ValueError, match="FSQ levels"):
        FiniteScalarQuantizer(levels)


@pytest.mark.parametrize(
    "build",
    [
        lambda: GroupedFiniteScalarQuantizer((8, 4), 0),
        lambda: ResidualFiniteScalarQuantizer((8, 4), 0, 0.0),
        lambda: ResidualVectorQuantizer(replace(RVQ_CFG, num_quantizers=0)),
    ],
    ids=["grouped_fsq", "residual_fsq", "rvq"],
)
def test_a_stack_without_quantizers_raises(build):
    with pytest.raises(ValueError, match="at least 1 quantizer"):
        build()


@pytest.mark.parametrize("tok", both_tokenizers())
@pytest.mark.parametrize("bad_index", ["dropped", "too_large"])
def test_decode_of_an_index_outside_the_codebook_raises(tok, bad_index):
    indices = torch.zeros(1, 4, 2, dtype=torch.long)
    indices[0, 1, 1] = DROPPED_CODE if bad_index == "dropped" else tok.codebook_size
    with pytest.raises(ValueError, match="outside 0"):
        tok.eval().decode(indices)


def test_an_rvq_tokenizer_without_its_config_raises():
    with pytest.raises(ValueError, match="needs its ResidualVectorQuantizationConfig"):
        build_tokenizer_network(replace(FSQ_CFG, kind=TokenizerKind.RESIDUAL_VECTOR))


@pytest.mark.parametrize("strict", [True, False])
def test_a_checkpoint_with_other_fsq_levels_of_the_same_length_is_refused(strict):
    trained = FiniteScalarQuantizationTokenizer(FSQ_CFG)
    other = FiniteScalarQuantizationTokenizer(replace(FSQ_CFG, fsq_levels=(8, 4, 8, 4)))
    with pytest.raises(RuntimeError, match="levels: the checkpoint has"):
        other.load_state_dict(trained.state_dict(), strict=strict)
