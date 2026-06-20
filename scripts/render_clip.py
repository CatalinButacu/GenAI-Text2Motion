"""Generate a streamed clip from a trained generator and render the recovered 22-joint skeleton as a
matplotlib 3D frame-strip PNG (headless, no aitviewer/GL needed) + save joints .npy. Lets us SEE the
motion and judge quality (recognizable vs broken) on any platform.

    $env:PYTHONPATH="src"; .venv/Scripts/python.exe scripts/render_clip.py --prompt "..." --cfg_scale 3.0
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.render.studio import kinematic_bones, recover_skeleton
from text2motion.shared.config import load_config
from text2motion.shared.seed import seed_everything
from text2motion.stream.decode import StreamingMotionDecoder


def run(a: argparse.Namespace) -> None:
    seed_everything(2026, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(a.config)
    tok = ResidualFsqTokenizer(cfg.tokenizer)
    tok.load_state_dict(torch.load(a.tokenizer_ckpt, map_location="cpu"))
    tok.to(dev).eval()
    nl = cfg.generator.mamba_n_layers if a.backbone == "mamba" else cfg.generator.n_layers
    gc = replace(cfg.generator, backbone=a.backbone, n_layers=nl,
                 num_codebooks=cfg.tokenizer.num_quantizers, codebook_size=tok.codebook_size,
                 use_kernel=False)
    gen = MotionGenerator(gc).to(dev).eval()
    st = torch.load(a.ckpt, map_location="cpu")
    gen.load_state_dict(st["generator"] if isinstance(st, dict) and "generator" in st else st)
    te = CLIPTextEncoder(cfg.text_encoder).to(dev).eval()
    with torch.no_grad():
        emb = te([a.prompt]).to(dev)
    out = Path(cfg.paths.hml3d_out_dir)
    mean = torch.from_numpy(np.load(out / "Mean.npy").astype("float32")).to(dev)
    std = torch.from_numpy(np.load(out / "Std.npy").astype("float32")).to(dev)
    dec = StreamingMotionDecoder(tok, mean, std)
    it = gen.stream(emb, a.steps, temperature=a.temperature, top_p=a.top_p,
                    cfg_scale=a.cfg_scale, stop_at_end=not a.fixed_length)
    chunks = []
    with torch.no_grad():
        for ch in dec.stream_tokens(it):
            chunks.append(ch.squeeze(0).cpu().numpy())
    feat = np.concatenate(chunks, 0) if chunks else np.zeros((0, 263), "float32")
    joints = recover_skeleton(feat)  # (T, 22, 3), HumanML3D y-up
    np.save(a.npy, joints)
    print(f"frames {joints.shape[0]}  -> saved {a.npy}")
    if joints.shape[0] == 0:
        print("NO FRAMES generated"); return

    bones = kinematic_bones()
    T = joints.shape[0]
    idx = np.linspace(0, T - 1, min(a.frames, T)).astype(int)
    # vertical = HumanML3D y; plot axes as (x, z, y) so the figure stands upright.
    lo, hi = joints.min((0, 1)), joints.max((0, 1))
    fig = plt.figure(figsize=(2.6 * len(idx), 3.2))
    for k, fi in enumerate(idx):
        ax = fig.add_subplot(1, len(idx), k + 1, projection="3d")
        p = joints[fi]
        for b in bones:
            ax.plot([p[b[0], 0], p[b[1], 0]], [p[b[0], 2], p[b[1], 2]], [p[b[0], 1], p[b[1], 1]],
                    color="tab:blue", lw=2)
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=6, color="tab:red")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_zlim(lo[1], hi[1])
        ax.set_title(f"t={fi}", fontsize=9); ax.set_axis_off(); ax.view_init(elev=12, azim=-75)
    fig.suptitle(f'{a.backbone} pilot, cfg {a.cfg_scale}: "{a.prompt}"  ({T} frames @20fps)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


def main() -> None:
    p = argparse.ArgumentParser(description="Render a streamed generated clip to a skeleton PNG strip.")
    p.add_argument("--config", default="configs/gen_pilot_fsq8x1024.yaml")
    p.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    p.add_argument("--ckpt", default="checkpoints/generator_transformer.pt")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/fsq_g8_v1024.pt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int, default=49)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fixed_length", action="store_true")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--npy", default="outputs/demo_joints.npy")
    p.add_argument("--out", default="paper/figures/demo_clip.png")
    run(p.parse_args())


if __name__ == "__main__":
    main()
