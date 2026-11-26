import pytest
import torch

from text2motion.generation.contracts import Backbone, GeneratorConfig, StreamAnchors
from text2motion.generation.model.generator import MotionTokenGenerator

D_TEXT = 16
STEPS = 10000
EARLY_STEP = 100
ANCHORS = StreamAnchors(every=10, context=8)


def tiny_twin(backbone: Backbone) -> MotionTokenGenerator:
    cfg = GeneratorConfig(
        backbone=backbone,
        d_model=32,
        n_layers=2,
        d_text=D_TEXT,
        num_codebooks=2,
        codebook_size=16,
        max_seq_len=32,
        n_heads=4,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank=8,
    )
    return MotionTokenGenerator(cfg).eval()


def state_bytes(state) -> int:
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, (list, tuple)):
        return sum(state_bytes(item) for item in state)
    if state is None:
        return 0
    raise TypeError(f"unexpected stream state element {type(state).__name__}")


@pytest.mark.parametrize("backbone", [Backbone.MAMBA, Backbone.TRANSFORMER])
def test_stream_state_size_is_constant_over_ten_thousand_anchored_steps(backbone: Backbone):
    torch.manual_seed(2026)
    gen = tiny_twin(backbone)
    active = []
    build_step = gen.cuda_graph.decode_step_for

    def recording_step(state, batch, text_vector):
        step = build_step(state, batch, text_vector)
        active.append(step)
        return step

    gen.cuda_graph.decode_step_for = recording_step
    sizes = [
        state_bytes(active[-1].state)
        for _ in gen.stream_token_indices(torch.randn(1, D_TEXT), STEPS, anchors=ANCHORS)
    ]

    assert len(sizes) == STEPS
    assert sizes[EARLY_STEP - 1] > 0
    assert sizes[EARLY_STEP - 1] == sizes[STEPS - 1]
    assert max(sizes) == max(sizes[:EARLY_STEP])
