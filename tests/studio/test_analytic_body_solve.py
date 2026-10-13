from pathlib import Path

import numpy as np
import pytest
import torch

from text2motion.studio.avatar import (
    build_smplx_body_model,
    solve_smplx_pose_from_joints,
)
from text2motion.studio.config import SmplxFitConfig

MODEL_DIR = Path("data/smplx_models")


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
