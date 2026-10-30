import torch
import torch.nn.functional as F

from text2motion.generation.nn_backbones import DT_FLOOR, DT_MAX, MambaMixer


def test_a_fresh_mamba_mixer_starts_with_the_reference_dt_and_a_long_memory():
    torch.manual_seed(0)
    mixer = MambaMixer(d_model=32, d_state=16, d_conv=4, expand=2, dt_rank=8)

    dt = F.softplus(mixer.dt_proj.bias.detach())
    kept = torch.exp(dt[:, None] * -torch.exp(mixer.a_log.detach()))

    assert float(dt.min()) >= DT_FLOOR * 0.999
    assert float(dt.max()) <= DT_MAX * 1.001
    assert float(mixer.dt_proj.weight.detach().abs().max()) <= 8**-0.5
    assert float(kept.median()) > 0.5
