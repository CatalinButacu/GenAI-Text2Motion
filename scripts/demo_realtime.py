"""REAL-TIME streaming demo: watch motion appear AS the generator produces it.

A producer thread runs generator.stream (bounded recurrent state, CFG-guided) -> decode chunk-by-chunk
-> a bounded queue; the main thread's matplotlib 3D animation pulls frames off the queue and plays
them at 20 fps as they arrive. Playback STARTS before generation finishes -- that is the streaming
novelty, live and visible. Opens a window (needs a display).

    $env:PYTHONPATH="src"; .venv/Scripts/python.exe scripts/demo_realtime.py `
      --prompt "a person walks forward and sits down" --cfg_scale 3.0 --fixed_length

--headless verifies the streaming timing (frames arriving over time) without a window.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.render.studio import kinematic_bones, recover_skeleton
from text2motion.shared.config import load_config
from text2motion.shared.seed import seed_everything
from text2motion.stream.decode import StreamingMotionDecoder

SENTINEL = None


def load_pipeline(a, dev):
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
    out = Path(cfg.paths.hml3d_out_dir)
    mean = torch.from_numpy(np.load(out / "Mean.npy").astype("float32")).to(dev)
    std = torch.from_numpy(np.load(out / "Std.npy").astype("float32")).to(dev)
    return tok, gen, te, mean, std


def run(a: argparse.Namespace) -> None:
    if a.headless:
        matplotlib.use("Agg")  # explicit non-interactive backend; windowed mode uses matplotlib's
        # own default interactive backend (no try/except over GUI libraries -- fail loud if none).
    seed_everything(2026, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, gen, te, mean, std = load_pipeline(a, dev)
    q: "queue.Queue" = queue.Queue(maxsize=16)

    def producer():
        with torch.no_grad():
            emb = te([a.prompt]).to(dev)
            dec = StreamingMotionDecoder(tok, mean, std)
            it = gen.stream(emb, a.steps, temperature=a.temperature, top_p=a.top_p,
                            cfg_scale=a.cfg_scale, stop_at_end=not a.fixed_length)
            for chunk in dec.stream_tokens(it):
                q.put(recover_skeleton(chunk.squeeze(0).cpu().numpy()))  # (cf, 22, 3)
        q.put(SENTINEL)

    threading.Thread(target=producer, daemon=True).start()

    if a.headless:
        n, t0 = 0, time.time()
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            n += item.shape[0]
            print(f"  t={time.time() - t0:5.1f}s  {n:4d} frames generated (streaming live)")
        print(f"done: {n} frames in {time.time() - t0:.1f}s")
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames: list = []
    done = [False]

    def drain():
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is SENTINEL:
                done[0] = True
                break
            frames.extend(list(item))

    bones = kinematic_bones()
    limsrc = Path("outputs/demo_joints.npy")
    lim = np.load(limsrc) if limsrc.is_file() else None
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")
    play = [0]

    def update(_):
        drain()
        if not frames:
            return
        p = frames[play[0]] if play[0] < len(frames) else frames[-1]
        if play[0] < len(frames):
            play[0] += 1
        ax.clear()
        ax.set_axis_off()
        for b in bones:
            ax.plot([p[b[0], 0], p[b[1], 0]], [p[b[0], 2], p[b[1], 2]], [p[b[0], 1], p[b[1], 1]],
                    color="tab:blue", lw=2)
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=8, color="tab:red")
        if lim is not None:
            lo, hi = lim.min((0, 1)), lim.max((0, 1))
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_zlim(lo[1], hi[1])
        ax.view_init(elev=12, azim=-75)
        tag = "generating..." if not done[0] else "done"
        ax.set_title(f'"{a.prompt}"\nframe {play[0]}/{len(frames)}  ({tag})', fontsize=9)

    FuncAnimation(fig, update, interval=50, cache_frame_data=False)  # ~20 fps
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time streaming motion demo (live 3D animation).")
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
    p.add_argument("--headless", action="store_true", help="verify streaming timing, no window")
    run(p.parse_args())


if __name__ == "__main__":
    main()
