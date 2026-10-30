import math

import pytest
import torch
import torch.nn.functional as F

from text2motion.generation.contracts import Backbone, GeneratorConfig
from text2motion.generation.nn_backbones import (
    DT_FLOOR,
    DT_MAX,
    MambaBlock,
    MambaMixer,
    TransformerBlock,
)
from text2motion.generation.nn_model import MotionTokenGenerator


def test_a_fresh_mamba_mixer_starts_with_the_reference_dt_and_a_long_memory():
    torch.manual_seed(0)
    mixer = MambaMixer(d_model=32, d_state=16, d_conv=4, expand=2, dt_rank=8)

    dt = F.softplus(mixer.dt_proj.bias.detach())
    kept = torch.exp(dt[:, None] * -torch.exp(mixer.a_log.detach()))

    assert float(dt.min()) >= DT_FLOOR * 0.999
    assert float(dt.max()) <= DT_MAX * 1.001
    assert float(mixer.dt_proj.weight.detach().abs().max()) <= 8**-0.5
    assert float(kept.median()) > 0.5


def __cfg(backbone: Backbone) -> GeneratorConfig:
    return GeneratorConfig(backbone=backbone, d_model=128, n_layers=4, n_heads=4, codebook_size=512)


def __residual_outputs(model: MotionTokenGenerator) -> list[tuple[torch.Tensor, int]]:
    layers = []
    for module in model.modules():
        if isinstance(module, MambaBlock):
            layers.append((module.mixer.out_proj.weight, 1))
        if isinstance(module, TransformerBlock):
            layers.append((module.proj.weight, 2))
            layers.append((module.mlp[2].weight, 2))
    return layers


@pytest.mark.parametrize("backbone", list(Backbone))
def test_both_twins_start_from_the_mamba_ssm_reference_init(backbone):
    torch.manual_seed(0)
    cfg = __cfg(backbone)
    model = MotionTokenGenerator(cfg)

    embeddings = [
        module.weight for module in model.modules() if isinstance(module, torch.nn.Embedding)
    ]
    for weight in embeddings:
        assert abs(float(weight.detach().std()) - 0.02) < 0.002

    for weight, residuals_per_layer in __residual_outputs(model):
        bound = 1 / math.sqrt(weight.size(1) * residuals_per_layer * cfg.n_layers)
        assert float(weight.detach().abs().max()) <= bound * 1.0001

    for name, parameter in model.named_parameters():
        if name.endswith(".bias") and "dt_proj" not in name and "conv1d" not in name:
            assert float(parameter.detach().abs().max()) == 0, name
