from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import smplx
import torch

from text2motion.motion.contracts import SMPLX_MODEL_TYPE
from text2motion.motion.representation import JOINTS
from text2motion.studio.contracts import SmplxFitResult
from text2motion.studio.studio_config import SmplxFitConfig

logger = logging.getLogger(__name__)


def build_smplx_body_model(cfg: SmplxFitConfig, t: int, device: str):
    return smplx.create(
        cfg.model_dir,
        model_type=SMPLX_MODEL_TYPE,
        gender=cfg.gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=cfg.num_betas,
        ext="npz",
        batch_size=t,
    ).to(device)


def yaw_init(
    model,
    target: torch.Tensor,
    transl: torch.Tensor,
    device: str,
    candidates_deg: tuple[float, ...],
) -> torch.Tensor:
    t = target.shape[0]
    best_aa, best_err = None, float("inf")
    with torch.no_grad():
        for yaw in np.deg2rad(candidates_deg):
            aa = torch.zeros(t, 3, device=device)
            aa[:, 1] = yaw
            out = model(
                global_orient=aa,
                body_pose=torch.zeros(t, 63, device=device),
                transl=transl,
                betas=torch.zeros(t, model.num_betas, device=device),
            )
            err = (out.joints[:, :JOINTS] - target).pow(2).sum(-1).mean().item()
            if err < best_err:
                best_err, best_aa = err, aa.clone()
    if best_aa is None:
        raise ValueError("yaw initialisation found no candidate; candidates_deg must be non-empty")
    return best_aa


def start_parameters(
    model,
    target: torch.Tensor,
    cfg: SmplxFitConfig,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = target.shape[0]
    if warm_start is None:
        root_position = target[:, 0].detach()
        yaw = yaw_init(model, target, root_position, device, cfg.yaw_candidates_deg)
        body_pose = torch.zeros(frames, 63, device=device, requires_grad=True)
        betas = torch.zeros(1, cfg.num_betas, device=device, requires_grad=True)
        return yaw.requires_grad_(True), body_pose, betas
    last_orient, last_pose, last_betas = warm_start
    orient = torch.tensor(last_orient, dtype=torch.float32, device=device).expand(frames, -1)
    pose = torch.tensor(last_pose, dtype=torch.float32, device=device).expand(frames, -1)
    betas = torch.tensor(last_betas, dtype=torch.float32, device=device).unsqueeze(0)
    return orient.clone().requires_grad_(True), pose.clone().requires_grad_(True), betas


def is_report_step(step: int, steps: int, every: int) -> bool:
    return step % every == 0 or step == steps - 1


def create_smplx_rest_pose_mesh(
    cfg: SmplxFitConfig, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    model = build_smplx_body_model(cfg, 1, device)
    with torch.no_grad():
        out = model(
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            transl=torch.zeros(1, 3, device=device),
            betas=torch.zeros(1, cfg.num_betas, device=device),
        )
    return out.vertices[0].cpu().numpy(), model.faces.astype(np.int64)


def smplx_mesh_from_fit_parameters(
    cfg: SmplxFitConfig,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray,
    gender: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    t = global_orient.shape[0]
    model = build_smplx_body_model(replace(cfg, gender=gender), t, device)
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(global_orient, device=device),
            body_pose=torch.tensor(body_pose, device=device),
            transl=torch.tensor(transl, device=device),
            betas=torch.tensor(betas, dtype=torch.float32, device=device).expand(t, -1),
        )
    return out.vertices.cpu().numpy(), model.faces.astype(np.int64)


def fit_forward(model, global_orient, body_pose, transl, betas) -> torch.Tensor:
    return model(
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        betas=betas.expand(global_orient.shape[0], -1),
    ).joints[:, :JOINTS]


def joint_error_cm(joints: torch.Tensor, target: torch.Tensor) -> float:
    return (joints - target).norm(dim=-1).mean().item() * 100


def run_stage_one(model, target, params, cfg, on_progress, total_iters) -> None:
    global_orient, body_pose, transl, betas = params
    stage1 = torch.optim.Adam([global_orient, transl], lr=cfg.lr, weight_decay=0.0)
    for it in range(cfg.stage1_iters):
        stage1.zero_grad()
        joints = fit_forward(model, global_orient, body_pose, transl, betas)
        ((joints - target) ** 2).sum(-1).mean().backward()
        stage1.step()
        if on_progress is not None and is_report_step(it, cfg.stage1_iters, cfg.progress_every):
            on_progress(it, total_iters)


def stage_two_loss(joints, target, global_orient, body_pose, cfg) -> torch.Tensor:
    data = ((joints - target) ** 2).sum(-1).mean()
    smooth = (body_pose[1:] - body_pose[:-1]).pow(2).mean() + (
        global_orient[1:] - global_orient[:-1]
    ).pow(2).mean()
    return data + cfg.w_smooth * smooth + cfg.w_reg * body_pose.pow(2).mean()


def run_stage_two(
    model, target, params, cfg, warm_start, verbose, on_progress, total_iters
) -> None:
    global_orient, body_pose, transl, betas = params
    stage2_params = [global_orient, body_pose, transl] + ([] if warm_start is not None else [betas])
    stage2 = torch.optim.Adam(stage2_params, lr=cfg.lr, weight_decay=0.0)
    for it in range(cfg.stage2_iters):
        stage2.zero_grad()
        joints = fit_forward(model, global_orient, body_pose, transl, betas)
        loss = stage_two_loss(joints, target, global_orient, body_pose, cfg)
        loss.backward()
        stage2.step()
        if on_progress is not None and is_report_step(it, cfg.stage2_iters, cfg.progress_every):
            on_progress(cfg.stage1_iters + it, total_iters)
        if verbose and is_report_step(it, cfg.stage2_iters, cfg.verbose_every):
            err = joint_error_cm(joints.detach(), target)
            logger.info(f"  fit it {it:3d}  joint err {err:.1f} cm")


def build_fit_result(model, target, params) -> SmplxFitResult:
    global_orient, body_pose, transl, betas = params
    with torch.no_grad():
        out = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas.expand(target.shape[0], -1),
        )
        err_cm = joint_error_cm(out.joints[:, :JOINTS], target)
    return SmplxFitResult(
        vertices=out.vertices.cpu().numpy(),
        faces=model.faces.astype(np.int64),
        global_orient=global_orient.detach().cpu().numpy(),
        body_pose=body_pose.detach().cpu().numpy(),
        transl=transl.detach().cpu().numpy(),
        betas=betas.detach().cpu().numpy()[0],
        joint_err_cm=err_cm,
    )


def fit_smplx_to_joints(
    target_joints: np.ndarray,
    cfg: SmplxFitConfig,
    device: str = "cuda",
    verbose: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    model=None,
) -> SmplxFitResult:
    dev = device if torch.cuda.is_available() else "cpu"
    t = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=dev)
    if model is None:
        model = build_smplx_body_model(cfg, t, dev)
    total_iters = max(1, cfg.stage1_iters + cfg.stage2_iters)
    transl = target[:, 0].clone().requires_grad_(True)
    global_orient, body_pose, betas = start_parameters(model, target, cfg, warm_start, dev)
    params = (global_orient, body_pose, transl, betas)
    run_stage_one(model, target, params, cfg, on_progress, total_iters)
    run_stage_two(model, target, params, cfg, warm_start, verbose, on_progress, total_iters)
    return build_fit_result(model, target, params)
