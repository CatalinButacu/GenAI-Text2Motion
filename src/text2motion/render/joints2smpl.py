"""Fit SMPL-X parameters to recovered 22-joint positions (the field-standard joints2smpl path).

Our 263 representation only yields joint POSITIONS cleanly (``recover_from_ric``); its rot6d are in
the t2m-skeleton frame, not SMPL's rest pose, so feeding them straight to SMPL distorts the body
(measured ~27 cm joint error). Instead we optimise SMPL-X (global_orient, body_pose, transl, shared
betas) so the model's body joints match the target positions -- the same approach MDM / T2M-GPT /
MoMask use for mesh visualisation. Optimisation (not real-time): seconds per clip.

The 263 track is body-only, so hands/face stay neutral; only the 22 body joints are fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import smplx
import torch

J = 22  # SMPL-X body joints shared with the HumanML3D-22 skeleton (same ordering)
_YAW_CANDIDATES = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)  # coarse facing init -> avoid 180deg flips


@dataclass(frozen=True)
class FitConfig:
    model_dir: str  # dir containing smplx/SMPLX_NEUTRAL.npz
    gender: str = "neutral"
    num_betas: int = 10
    stage1_iters: int = 150  # orient + translation only
    stage2_iters: int = 300  # full pose + betas
    lr: float = 0.05
    w_smooth: float = 0.05  # temporal pose smoothness
    w_reg: float = 5e-4  # weak pose L2 (keeps elbows/knees from hyper-extending)


@dataclass
class FitResult:
    vertices: np.ndarray  # (T, V, 3)
    faces: np.ndarray  # (F, 3)
    global_orient: np.ndarray  # (T, 3) axis-angle
    body_pose: np.ndarray  # (T, 63)
    transl: np.ndarray  # (T, 3)
    betas: np.ndarray  # (num_betas,)
    joint_err_cm: float  # final mean joint error vs target


def _build_model(cfg: FitConfig, t: int, device: str):
    return smplx.create(
        cfg.model_dir,
        model_type="smplx",
        gender=cfg.gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=cfg.num_betas,
        ext="npz",
        batch_size=t,
    ).to(device)


def _yaw_init(model, target: torch.Tensor, transl: torch.Tensor, device: str) -> torch.Tensor:
    """Pick the frame-0 yaw (of 4) whose zero-pose SMPL joints best match the target -> init orient."""
    t = target.shape[0]
    best_aa, best_err = None, float("inf")
    with torch.no_grad():
        for yaw in _YAW_CANDIDATES:
            aa = torch.zeros(t, 3, device=device)
            aa[:, 1] = yaw  # rotate about the world up (y) axis
            out = model(
                global_orient=aa,
                body_pose=torch.zeros(t, 63, device=device),
                transl=transl,
                betas=torch.zeros(t, model.num_betas, device=device),
            )
            err = (out.joints[:, :J] - target).pow(2).sum(-1).mean().item()
            if err < best_err:
                best_err, best_aa = err, aa.clone()
    return best_aa


def fit_smplx_to_joints(
    target_joints: np.ndarray, cfg: FitConfig, device: str = "cuda", verbose: bool = False
) -> FitResult:
    """(T, 22, 3) target positions -> fitted SMPL-X. Two-stage Adam (orient/transl, then full)."""
    dev = device if torch.cuda.is_available() else "cpu"
    t = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=dev)
    model = _build_model(cfg, t, dev)

    transl = target[:, 0].clone().requires_grad_(True)  # init at the pelvis target
    global_orient = _yaw_init(model, target, target[:, 0].detach(), dev).requires_grad_(True)
    body_pose = torch.zeros(t, 63, device=dev, requires_grad=True)
    betas = torch.zeros(1, cfg.num_betas, device=dev, requires_grad=True)

    def forward():
        return model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas.expand(t, -1),
        ).joints[:, :J]

    stage1 = torch.optim.Adam([global_orient, transl], lr=cfg.lr)
    for _ in range(cfg.stage1_iters):
        stage1.zero_grad()
        ((forward() - target) ** 2).sum(-1).mean().backward()
        stage1.step()

    stage2 = torch.optim.Adam([global_orient, body_pose, transl, betas], lr=cfg.lr)
    for it in range(cfg.stage2_iters):
        stage2.zero_grad()
        joints = forward()
        data = ((joints - target) ** 2).sum(-1).mean()
        smooth = (body_pose[1:] - body_pose[:-1]).pow(2).mean() + (
            global_orient[1:] - global_orient[:-1]
        ).pow(2).mean()
        loss = data + cfg.w_smooth * smooth + cfg.w_reg * body_pose.pow(2).mean()
        loss.backward()
        stage2.step()
        if verbose and (it % 100 == 0 or it == cfg.stage2_iters - 1):
            err = (joints.detach() - target).norm(dim=-1).mean().item() * 100
            print(f"  fit it {it:3d}  joint err {err:.1f} cm")

    with torch.no_grad():
        out = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas.expand(t, -1),
        )
        err_cm = (out.joints[:, :J] - target).norm(dim=-1).mean().item() * 100
    return FitResult(
        vertices=out.vertices.cpu().numpy(),
        faces=model.faces.astype(np.int64),
        global_orient=global_orient.detach().cpu().numpy(),
        body_pose=body_pose.detach().cpu().numpy(),
        transl=transl.detach().cpu().numpy(),
        betas=betas.detach().cpu().numpy()[0],
        joint_err_cm=err_cm,
    )
