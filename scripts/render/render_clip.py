# python scripts/render/render_clip.py --prompt "a person walks forward" --out artifacts/outputs/figures/demo_clip.png
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.commands.serving import load_inference_pipeline
from text2motion.generation.contracts import Backbone
from text2motion.motion.representation import DIM, FPS, kinematic_bones, recover_skeleton
from text2motion.streaming.pydant_requests import ModelServiceRequest


def run(a: argparse.Namespace) -> None:
    app = ApplicationBootstrap.from_config(a.config)
    cfg = app.config
    steps = a.steps if a.steps is not None else cfg.data.max_motion_len // cfg.tokenizer.downsample
    request = ModelServiceRequest(
        config=Path(a.config),
        ckpt=Path(a.ckpt),
        tokenizer_ckpt=Path(a.tokenizer_ckpt),
        backbone=Backbone(a.backbone),
    )
    pipeline = load_inference_pipeline(app, request)
    chunks = []
    with torch.no_grad():
        emb = pipeline.generator.text_encoder([a.prompt]).to(pipeline.device)
        it = pipeline.generator.token_generator.stream_token_indices(
            emb,
            steps,
            temperature=a.temperature,
            top_p=a.top_p,
            cfg_scale=a.cfg_scale,
            stop_at_end=not a.fixed_length,
        )
        for ch in pipeline.decoder.stream_tokens(it):
            chunks.append(ch.squeeze(0).cpu().numpy())
    feat = np.concatenate(chunks, 0) if chunks else np.zeros((0, DIM), "float32")
    joints = recover_skeleton(feat)
    np.save(a.npy, joints)
    print(f"frames {joints.shape[0]}  -> saved {a.npy}")
    if joints.shape[0] == 0:
        print("NO FRAMES generated")
        return

    bones = kinematic_bones()
    frame_count = joints.shape[0]
    idx = np.linspace(0, frame_count - 1, min(a.frames, frame_count)).astype(int)

    lo, hi = joints.min((0, 1)), joints.max((0, 1))
    fig = plt.figure(figsize=(2.6 * len(idx), 3.2))
    for k, fi in enumerate(idx):
        ax = fig.add_subplot(1, len(idx), k + 1, projection="3d")
        if not isinstance(ax, Axes3D):
            raise TypeError("the 3d projection did not give 3D axes")
        p = joints[fi]
        for b in bones:
            ax.plot(
                [p[b[0], 0], p[b[1], 0]],
                [p[b[0], 2], p[b[1], 2]],
                [p[b[0], 1], p[b[1], 1]],
                color="tab:blue",
                lw=2,
            )
        ax.plot(
            p[:, 0], p[:, 2], p[:, 1], linestyle="", marker="o", markersize=2.5, color="tab:red"
        )
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[2], hi[2])
        ax.set_zlim(lo[1], hi[1])
        ax.set_title(f"t={fi}", fontsize=9)
        ax.set_axis_off()
        ax.view_init(elev=12, azim=-75)
    fig.suptitle(
        f'{a.backbone} pilot, cfg {a.cfg_scale}: "{a.prompt}"  ({frame_count} frames @{FPS}fps)',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a streamed generated clip to a skeleton PNG strip."
    )
    p.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    p.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    p.add_argument("--ckpt", default="artifacts/checkpoints/generator/generator_transformer.pt")
    p.add_argument("--tokenizer_ckpt", default="artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fixed_length", action="store_true")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--npy", default="artifacts/outputs/demo_joints.npy")
    p.add_argument("--out", default="artifacts/outputs/figures/demo_clip.png")
    run(p.parse_args())


if __name__ == "__main__":
    main()
