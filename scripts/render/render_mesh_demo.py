\
\
\
\
\
\
\
\
   

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.render.joints2smpl import FitConfig, fit_smplx_to_joints
from text2motion.render.studio import recover_skeleton
from text2motion.shared.config import load_config
from text2motion.shared.seed import seed_everything
from text2motion.stream.decode import StreamingMotionDecoder

MODEL_DIR = r"D:\Facultate\dissertation\data\arctic\unpack\models"


def draw(ax, verts, faces, frame, prompt):
    v = verts[frame]
    ax.clear()
    ax.plot_trisurf(v[:, 0], v[:, 2], v[:, 1], triangles=faces, color=(0.55, 0.6, 0.9),
                    edgecolor="none", linewidth=0)
    c = verts.reshape(-1, 3).mean(0)
    ax.set_xlim(c[0] - 1, c[0] + 1)
    ax.set_ylim(c[2] - 1, c[2] + 1)
    ax.set_zlim(0, 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=8, azim=-70)
    ax.set_axis_off()


def run(a):
    seed_everything(2026, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(a.config)
    tok = ResidualFsqTokenizer(cfg.tokenizer)
    tok.load_state_dict(torch.load(a.tokenizer_ckpt, map_location="cpu"))
    tok.to(dev).eval()
    gc = replace(cfg.generator, backbone="transformer", n_layers=cfg.generator.n_layers,
                 num_codebooks=cfg.tokenizer.num_quantizers, codebook_size=tok.codebook_size,
                 use_kernel=False)
    gen = MotionGenerator(gc).to(dev).eval()
    st = torch.load(a.ckpt, map_location="cpu")
    gen.load_state_dict(st["generator"] if isinstance(st, dict) and "generator" in st else st)
    te = CLIPTextEncoder(cfg.text_encoder).to(dev).eval()
    out = Path(cfg.paths.hml3d_out_dir)
    mean = torch.from_numpy(np.load(out / "Mean.npy").astype("float32")).to(dev)
    std = torch.from_numpy(np.load(out / "Std.npy").astype("float32")).to(dev)

    with torch.no_grad():
        emb = te([a.prompt]).to(dev)
        dec = StreamingMotionDecoder(tok, mean, std)
        it = gen.stream(emb, a.steps, temperature=a.temperature, top_p=a.top_p,
                        cfg_scale=a.cfg_scale, stop_at_end=False)
        joints = np.concatenate(
            [recover_skeleton(c.squeeze(0).cpu().numpy()) for c in dec.stream_tokens(it)], axis=0
        )
    print(f"[{a.tag}] generated {joints.shape[0]} frames; fitting SMPL-X...")
    if dev == "cuda":
        torch.cuda.empty_cache()
    res = fit_smplx_to_joints(joints, FitConfig(model_dir=MODEL_DIR), device=dev)
    print(f"[{a.tag}] fit done: {res.joint_err_cm:.1f} cm joint error")
    verts, faces = res.vertices, res.faces

                                                                       
    act_dir = Path("outputs/demo_gallery") / a.tag
    act_dir.mkdir(parents=True, exist_ok=True)
    (act_dir / "prompt.txt").write_text(a.prompt, encoding="utf-8")

                                     
    idx = np.linspace(0, len(verts) - 1, 6).astype(int)
    fig = plt.figure(figsize=(18, 3.4))
    for i, f in enumerate(idx):
        ax = fig.add_subplot(1, 6, i + 1, projection="3d")
        draw(ax, verts, faces, f, a.prompt)
        ax.set_title(f"f{f}", fontsize=8)
    fig.suptitle(f'"{a.prompt}"   (SMPL-X body, fit {res.joint_err_cm:.1f}cm)', fontsize=11)
    fig.tight_layout()
    montage = str(act_dir / f"{a.tag}.png")
    fig.savefig(montage, dpi=70)
    plt.close(fig)

                  
    figg = plt.figure(figsize=(4, 5))
    axg = figg.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=a.fps)
    gif = str(act_dir / f"{a.tag}.gif")
    with writer.saving(figg, gif, dpi=60):
        for f in range(len(verts)):
            draw(axg, verts, faces, f, a.prompt)
            axg.set_title(f'"{a.prompt}"  f{f + 1}/{len(verts)}', fontsize=9)
            writer.grab_frame()
    plt.close(figg)
    print(f"[{a.tag}] saved {montage} + {gif}")


def main():
    p = argparse.ArgumentParser(description="Render generated motion as a SMPL-X body montage + GIF.")
    p.add_argument("--config", default="configs/generator/final100m_fsq8x1024.yaml")
    p.add_argument("--ckpt", default="checkpoints/generator/generator_transformer_100m_val163.pt")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=49)
    p.add_argument("--cfg_scale", type=float, default=6.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fps", type=int, default=20)
    run(p.parse_args())


if __name__ == "__main__":
    main()
