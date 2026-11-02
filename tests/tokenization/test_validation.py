from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from text2motion.motion.contracts import MotionBatch, MotionClip, MotionTokens
from text2motion.motion.normalization import MotionScaler
from text2motion.tokenization.contracts import (
    ResidualVectorQuantizationConfig,
    TokenizerConfig,
    TokenizeRequest,
)
from text2motion.tokenization.corpus import build_token_pack
from text2motion.tokenization.nn_model import (
    FiniteScalarQuantizationTokenizer,
    FrozenMotionTokenizer,
    ResidualVectorQuantizationTokenizer,
    feature_velocity_reconstruction_loss,
)
from text2motion.tokenization.nn_quantizers import FiniteScalarQuantizer, VectorQuantizer
from text2motion.tokenization.trainer import TokenizerTrainer

FSQ = TokenizerConfig(width=32, num_quantizers=2, fsq_levels=(4, 4), n_resblocks=1)
RVQ = ResidualVectorQuantizationConfig(
    width=32, code_dim=8, codebook_size=16, num_quantizers=2, n_resblocks=1, quant_dropout=0.0
)


def __frozen() -> FrozenMotionTokenizer:
    torch.manual_seed(0)
    return FrozenMotionTokenizer(FiniteScalarQuantizationTokenizer(FSQ))


def test_float_token_indices_are_refused():
    with pytest.raises(TypeError, match="integer tensor"):
        FiniteScalarQuantizationTokenizer(FSQ).decode(torch.zeros(1, 4, 2))


def test_fsq_levels_with_an_unreachable_end_digit_are_refused():
    with pytest.raises(ValueError, match="unreachable"):
        FiniteScalarQuantizer((1024,))


def test_the_frozen_decoder_refuses_a_batch_and_a_wrong_frame_count():
    frozen = __frozen()
    with pytest.raises(ValueError, match="batch of 2"):
        frozen.decode(MotionTokens(indices=torch.zeros(2, 4, 2, dtype=torch.long), frame_count=16))
    with pytest.raises(ValueError, match="the tokens say 20"):
        frozen.decode(MotionTokens(indices=torch.zeros(4, 2, dtype=torch.long), frame_count=20))


def test_the_frozen_encoder_refuses_fewer_feature_rows_than_frames():
    with pytest.raises(ValueError, match="says 64 frames"):
        __frozen().encode(MotionClip(features=torch.randn(20, 263), frame_count=64))


@pytest.mark.parametrize(
    ("codebook_size", "segment", "match"),
    [(40000, 16, "int16"), (16, 6, "multiple of the downsample")],
    ids=["too_many_codes", "unaligned_segment"],
)
def test_the_token_pack_refuses_overflow_and_unaligned_windows(
    tmp_path, codebook_size, segment, match
):
    features = tmp_path / "features"
    features.mkdir()
    np.save(features / "clip.npy", np.zeros((40, 263), np.float32))
    request = TokenizeRequest(
        config=tmp_path / "unused.yaml",
        features_dir=features,
        token_pack=tmp_path / "pack.npz",
        segment_frames=segment,
        stride=segment,
    )
    tokenizer = SimpleNamespace(codebook_size=codebook_size, downsample=4)
    scaler = cast(MotionScaler, SimpleNamespace(normalize=lambda feat: feat))
    with pytest.raises(ValueError, match=match):
        build_token_pack(cast(FrozenMotionTokenizer, tokenizer), scaler, request, set(), {})


def test_a_padded_batch_is_refused_by_the_tokenizer_trainer():
    trainer = TokenizerTrainer(
        FiniteScalarQuantizationTokenizer(FSQ), lr=1e-3, weight_decay=0.0, ema_decay=0.9
    )
    batch = MotionBatch(
        features=torch.zeros(2, 16, 263), lengths=torch.tensor([16, 8]), captions=["a", "b"]
    )
    with pytest.raises(ValueError, match="padded"):
        trainer.train_epoch([batch], "cpu")


def test_a_direct_forward_with_nan_leaves_the_rvq_codebooks_unchanged():
    network = ResidualVectorQuantizationTokenizer(RVQ).train()
    before = {
        name: value.clone() for name, value in network.state_dict().items() if "embed" in name
    }
    motion = torch.randn(1, 16, 263)
    motion[0, 3, 5] = float("nan")

    with pytest.raises(ValueError, match="NaN or Inf"):
        network(motion)

    for name, value in before.items():
        assert torch.equal(network.state_dict()[name], value), name


@pytest.mark.parametrize(
    "build",
    [
        lambda: replace(FSQ, quant_dropout=1.5),
        lambda: replace(FSQ, n_resblocks=-1),
        lambda: replace(RVQ, codebook_size=0),
        lambda: replace(RVQ, ema_decay=1.2),
    ],
    ids=["dropout", "resblocks", "codebook", "decay"],
)
def test_tokenizer_configs_refuse_values_outside_their_range(build):
    with pytest.raises(ValueError):
        build()


def test_the_reconstruction_loss_needs_two_frames():
    with pytest.raises(ValueError, match="at least 2 frames"):
        feature_velocity_reconstruction_loss(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3))


def test_rvq_returns_the_codes_of_the_codebook_used_for_the_assignment():
    quantizer = VectorQuantizer(codebook_size=2, code_width=1, ema_decay=0.0, reset_threshold=0.0)
    quantizer.embed.copy_(torch.tensor([[0.0], [10.0]]))
    quantizer.train()

    quantized, indices, _ = quantizer(torch.tensor([[[4.0], [6.0]]]))

    assert indices.tolist() == [[0, 1]]
    assert quantized.flatten().tolist() == [0.0, 10.0]
    assert torch.equal(quantizer.cluster_size, torch.tensor([1.0, 1.0]))


def test_the_rvq_counts_start_coherent_with_the_codebook():
    quantizer = VectorQuantizer(codebook_size=4, code_width=2, ema_decay=0.9, reset_threshold=0.0)

    assert torch.equal(quantizer.cluster_size, torch.ones(4))
    assert torch.equal(quantizer.embed_avg, quantizer.embed)


def test_a_resume_with_other_trainer_settings_is_refused():
    network = FiniteScalarQuantizationTokenizer(FSQ)
    first = TokenizerTrainer(network, lr=1e-3, weight_decay=0.0, ema_decay=0.9)
    state = first.resume_state(0, 1.0, None)
    second = TokenizerTrainer(network, lr=1e-3, weight_decay=0.0, ema_decay=0.5)

    with pytest.raises(ValueError, match="same settings"):
        second.load_resume_state(state, False)
