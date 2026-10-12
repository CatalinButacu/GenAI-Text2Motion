from pathlib import Path

import numpy as np
import pytest
import torch

from text2motion.studio.avatar import (
    build_smplx_body_model,
    rotation_matrix_to_axis_angle,
    solve_smplx_pose_from_joints,
)
from text2motion.studio.config import SmplxFitConfig

MODEL_DIR = Path("data/smplx_models")


def __axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = (axis_angle / angle).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).view(*x.shape, 3, 3)
    sine = torch.sin(angle)[..., None]
    cosine = torch.cos(angle)[..., None]
    return torch.eye(3) + sine * skew + (1 - cosine) * skew @ skew


def test_axis_angle_round_trips_including_near_half_turns():
    generator = torch.Generator().manual_seed(0)
    axes = torch.nn.functional.normalize(torch.randn(64, 3, generator=generator), dim=-1)
    angles = torch.cat(
        [torch.rand(60, generator=generator) * 3.0, torch.tensor([3.10, 3.13, 1e-4, 0.0])]
    )
    axis_angle = axes * angles[:, None]
    recovered = rotation_matrix_to_axis_angle(__axis_angle_to_matrix(axis_angle))
    assert torch.allclose(
        __axis_angle_to_matrix(recovered), __axis_angle_to_matrix(axis_angle), atol=1e-4
    )


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="SMPL-X model files not installed")
def test_analytic_solve_recovers_a_known_body_pose():
    frames = 8
    model = build_smplx_body_model(SmplxFitConfig(model_dir=str(MODEL_DIR)), frames, "cpu")
    generator = torch.Generator().manual_seed(1)
    body_pose = (torch.rand(frames, 63, generator=generator) - 0.5) * 0.8
    global_orient = torch.tensor([[0.0, 1.2, 0.0]]).repeat(frames, 1)
    transl = torch.tensor([[0.3, 0.9, -0.2]]).repeat(frames, 1)
    with torch.no_grad():
        target = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=torch.zeros(frames, model.num_betas),
        ).joints[:, :22]
    result = solve_smplx_pose_from_joints(model, target.numpy().astype(np.float32), "cpu")
    assert result.vertices.shape == (frames, 10475, 3)
    assert result.joint_err_cm < 1.5
