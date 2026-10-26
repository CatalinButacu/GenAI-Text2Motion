# python scripts/render/render_mesh_demo.py --prompt "a person waves" --tag wave --ckpt artifacts/checkpoints/generator/generator_transformer_bs8_e60.pt
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.commands.serving import load_inference_pipeline
from text2motion.generation.contracts import Backbone
from text2motion.motion.representation import FPS, StreamingSkeletonRecovery
from text2motion.streaming.pydant_requests import ModelServiceRequest
from text2motion.studio.avatar import fit_smplx_to_joints
from text2motion.studio.config import SmplxFitConfig


def draw(ax, verts, faces, frame):
    v = verts[frame]
    ax.clear()
    ax.plot_trisurf(
        v[:, 0],
        v[:, 2],
        v[:, 1],
        triangles=faces,
        color=(0.55, 0.6, 0.9),
        edgecolor="none",
        linewidth=0,
    )
    c = verts.reshape(-1, 3).mean(0)
    ax.set_xlim(c[0] - 1, c[0] + 1)
    ax.set_ylim(c[2] - 1, c[2] + 1)
    ax.set_zlim(0, 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=8, azim=-70)
    ax.set_axis_off()


def run(a):
    app = ApplicationBootstrap.from_config(a.config)
    cfg = app.config
    dev = app.device
    model_dir = Path(a.model_dir) if a.model_dir else cfg.paths.smplx_models
    if model_dir is None:
        raise ValueError("paths.smplx_models or --model_dir must identify the SMPL-X model root")
    steps = a.steps if a.steps is not None else cfg.data.max_motion_len // cfg.tokenizer.downsample
    fps = a.fps or FPS
    request = ModelServiceRequest(
        config=Path(a.config),
        ckpt=Path(a.ckpt),
        tokenizer_ckpt=Path(a.tokenizer_ckpt),
        backbone=Backbone(a.backbone),
    )
    pipeline = load_inference_pipeline(app, request)

    with torch.no_grad():
        emb = pipeline.generator.text_encoder([a.prompt]).to(pipeline.device)
        it = pipeline.generator.token_generator.stream_token_indices(
            emb,
            steps,
            temperature=a.temperature,
            top_p=a.top_p,
            cfg_scale=a.cfg_scale,
            stop_at_end=False,
        )
        recovery = StreamingSkeletonRecovery()
        joints = np.concatenate(
            [recovery(c.squeeze(0).cpu().numpy()) for c in pipeline.decoder.stream_tokens(it)],
            axis=0,
        )
    print(f"[{a.tag}] generated {joints.shape[0]} frames; fitting SMPL-X...")
    if dev == "cuda":
        torch.cuda.empty_cache()
    res = fit_smplx_to_joints(joints, SmplxFitConfig(model_dir=str(model_dir)), device=dev)
    print(f"[{a.tag}] fit done: {res.joint_err_cm:.1f} cm joint error")
    verts, faces = res.vertices, res.faces

    act_dir = Path(cfg.paths.outputs_dir) / "demo_gallery" / a.tag
    act_dir.mkdir(parents=True, exist_ok=True)
    (act_dir / "prompt.txt").write_text(a.prompt, encoding="utf-8")

    idx = np.linspace(0, len(verts) - 1, 6).astype(int)
    fig = plt.figure(figsize=(18, 3.4))
    for i, f in enumerate(idx):
        ax = fig.add_subplot(1, 6, i + 1, projection="3d")
        draw(ax, verts, faces, f)
        ax.set_title(f"f{f}", fontsize=8)
    fig.suptitle(f'"{a.prompt}"   (SMPL-X body, fit {res.joint_err_cm:.1f}cm)', fontsize=11)
    fig.tight_layout()
    montage = str(act_dir / f"{a.tag}.png")
    fig.savefig(montage, dpi=70)
    plt.close(fig)

    figg = plt.figure(figsize=(4, 5))
    axg = figg.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=fps)
    gif = str(act_dir / f"{a.tag}.gif")
    with writer.saving(figg, gif, dpi=60):
        for f in range(len(verts)):
            draw(axg, verts, faces, f)
            axg.set_title(f'"{a.prompt}"  f{f + 1}/{len(verts)}', fontsize=9)
            writer.grab_frame()
    plt.close(figg)
    print(f"[{a.tag}] saved {montage} + {gif}")


def main():
    p = argparse.ArgumentParser(
        description="Render generated motion as a SMPL-X body montage + GIF."
    )
    p.add_argument("--config", default="configs/generator/final100m_fsq8x1024.yaml")
    p.add_argument(
        "--ckpt", default="artifacts/checkpoints/generator/generator_transformer_100m_val163.pt"
    )
    p.add_argument("--tokenizer_ckpt", default="artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    p.add_argument("--model_dir")
    p.add_argument("--prompt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg_scale", type=float, default=6.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fps", type=int)
    run(p.parse_args())


if __name__ == "__main__":
    main()
