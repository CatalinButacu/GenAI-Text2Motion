from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

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


build_smplx_model = _build_model


def _yaw_init(model, target: torch.Tensor, transl: torch.Tensor, device: str) -> torch.Tensor:
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


def rest_pose_body(cfg: FitConfig, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model = _build_model(cfg, 1, device)
    with torch.no_grad():
        out = model(
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            transl=torch.zeros(1, 3, device=device),
            betas=torch.zeros(1, cfg.num_betas, device=device),
        )
    return out.vertices[0].cpu().numpy(), model.faces.astype(np.int64)


def mesh_from_params(
    cfg: FitConfig,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray,
    gender: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    t = global_orient.shape[0]
    model = _build_model(replace(cfg, gender=gender), t, device)
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(global_orient, device=device),
            body_pose=torch.tensor(body_pose, device=device),
            transl=torch.tensor(transl, device=device),
            betas=torch.tensor(betas, dtype=torch.float32, device=device).expand(t, -1),
        )
    return out.vertices.cpu().numpy(), model.faces.astype(np.int64)


def fit_smplx_to_joints(
    target_joints: np.ndarray,
    cfg: FitConfig,
    device: str = "cuda",
    verbose: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    model=None,
) -> FitResult:
    dev = device if torch.cuda.is_available() else "cpu"
    t = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=dev)
    if model is None:
        model = _build_model(cfg, t, dev)
    total_iters = max(1, cfg.stage1_iters + cfg.stage2_iters)

    transl = target[:, 0].clone().requires_grad_(True)  # init at the pelvis target
    if warm_start is not None:
        last_orient, last_pose, last_betas = warm_start
        global_orient = (
            torch.tensor(last_orient, dtype=torch.float32, device=dev)
            .expand(t, -1)
            .clone()
            .requires_grad_(True)
        )
        body_pose = (
            torch.tensor(last_pose, dtype=torch.float32, device=dev)
            .expand(t, -1)
            .clone()
            .requires_grad_(True)
        )
        betas = torch.tensor(last_betas, dtype=torch.float32, device=dev).unsqueeze(0)  # frozen
    else:
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

    if cfg.stage1_iters > 0:
        stage1 = torch.optim.Adam([global_orient, transl], lr=cfg.lr)
        for it in range(cfg.stage1_iters):
            stage1.zero_grad()
            ((forward() - target) ** 2).sum(-1).mean().backward()
            stage1.step()
            if on_progress is not None and (it % 10 == 0 or it == cfg.stage1_iters - 1):
                on_progress(it, total_iters)

    stage2_params = [global_orient, body_pose, transl] + ([] if warm_start is not None else [betas])
    stage2 = torch.optim.Adam(stage2_params, lr=cfg.lr)
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
        if on_progress is not None and (it % 10 == 0 or it == cfg.stage2_iters - 1):
            on_progress(cfg.stage1_iters + it, total_iters)
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
