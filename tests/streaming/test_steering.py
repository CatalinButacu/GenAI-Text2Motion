import pytest
import torch

from text2motion.generation.contracts import Backbone, GeneratorConfig
from text2motion.generation.decoder import SteeringMode, StreamingMotionDecoder, steer_stream
from text2motion.generation.nn_generator import MotionTokenGenerator
from text2motion.tokenization.contracts import TokenizerConfig
from text2motion.tokenization.nn_tokenizer import FiniteScalarQuantizationTokenizer

TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
    backbone=Backbone.MAMBA,
    d_model=32,
    n_layers=2,
    d_text=16,
    num_codebooks=2,
    codebook_size=16,
    max_seq_len=64,
    d_state=8,
    d_conv=4,
    expand=2,
    dt_rank=8,
)
STEPS = 24
SWITCH = 12
BLEND = 8


def steer(mode: SteeringMode) -> list[tuple[int, torch.Tensor]]:
    torch.manual_seed(2026)
    generator = MotionTokenGenerator(GEN).eval()
    decoder = StreamingMotionDecoder(
        FiniteScalarQuantizationTokenizer(TOK), 4, torch.zeros(263), torch.ones(263), chunk_tokens=4
    )
    old_text = torch.randn(1, 16)
    new_text = torch.randn(1, 16)
    return list(
        steer_stream(
            generator, decoder, old_text, new_text, STEPS, SWITCH, mode, blend_frames=BLEND
        )
    )


@pytest.mark.parametrize("mode", [SteeringMode.REANCHOR, SteeringMode.CROSSFADE])
def test_stream_continues_after_switch_without_frame_index_reset(mode: SteeringMode):
    chunks = steer(mode)
    expected = 0
    for frame_index, chunk in chunks:
        assert frame_index == expected
        expected += chunk.shape[1]
    assert expected == STEPS * TOK.downsample
    assert chunks[-1][0] + chunks[-1][1].shape[1] > SWITCH * TOK.downsample
    assert all(torch.isfinite(chunk).all() for _, chunk in chunks)
