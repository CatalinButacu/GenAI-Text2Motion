"""Real-time streaming demo: text prompt -> bounded-state streamed motion -> skeleton MP4 / viewer.

Loads a trained generator + the frozen FSQ tokenizer + CLIP, streams motion tokens chunk-by-chunk
with bounded recurrent state (generator.stream, CFG-guided), decodes each window to 263 frames as it
arrives (StreamingMotionDecoder), recovers the 22-joint skeleton, and renders an MP4 (headless) or
opens the interactive aitviewer studio. This is Contribution B made visible: motion appears as it is
generated, at fixed per-step cost.

Needs the viewer extra (install when the venv is free -- it is busy while a pilot trains):
    uv sync --extra viewer

    $env:PYTHONPATH="src"; .venv/Scripts/python.exe scripts/demo_stream.py `
      --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone transformer `
      --ckpt checkpoints/generator/generator_transformer.pt --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt `
      --prompt "a person walks forward and sits down" --steps 49 --out outputs/demo.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import load_config
from text2motion.shared.seed import seed_everything
from text2motion.stream.decode import StreamingMotionDecoder


def run(args: argparse.Namespace) -> None:
    seed_everything(2026, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config)

    tok = ResidualFsqTokenizer(cfg.tokenizer)
    tok.load_state_dict(torch.load(args.tokenizer_ckpt, map_location="cpu"))
    tok.to(dev).eval()

    n_layers = cfg.generator.mamba_n_layers if args.backbone == "mamba" else cfg.generator.n_layers
    gc = replace(
        cfg.generator,
        backbone=args.backbone,
        n_layers=n_layers,
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=tok.codebook_size,
        use_kernel=False,  # local demo uses the eager step path (no mamba-ssm needed)
    )
    gen = MotionGenerator(gc).to(dev).eval()
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state["generator"] if isinstance(state, dict) and "generator" in state else state
    gen.load_state_dict(sd)

    text_enc = CLIPTextEncoder(cfg.text_encoder).to(dev).eval()
    with torch.no_grad():
        text_emb = text_enc([args.prompt]).to(dev)

    out_dir = Path(cfg.paths.hml3d_out_dir)
    mean = torch.from_numpy(np.load(out_dir / "Mean.npy").astype(np.float32)).to(dev)
    std = torch.from_numpy(np.load(out_dir / "Std.npy").astype(np.float32)).to(dev)
    decoder = StreamingMotionDecoder(tok, mean, std)

    # Drive generator.stream directly so we get CFG guidance (the decoder helper omits cfg_scale),
    # then window the per-step tokens into frame chunks exactly as the live producer does.
    tok_iter = gen.stream(
        text_emb,
        args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        stop_at_end=not args.fixed_length,
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in decoder.stream_tokens(tok_iter):
            chunks.append(chunk.squeeze(0).cpu().numpy())  # (chunk_frames, 263)
            print(f"  streamed +{chunk.shape[1]} frames (total {sum(c.shape[0] for c in chunks)})")

    feat = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 263), np.float32)
    print(f"generated {feat.shape[0]} frames ~ {feat.shape[0] / 20:.1f}s @ 20fps for: {args.prompt!r}")

    from text2motion.render.studio import recover_skeleton, render_skeleton_video, view_skeleton

    joints = recover_skeleton(feat)  # (T, 22, 3)
    if args.interactive:
        view_skeleton(joints)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        render_skeleton_video(joints, args.out)
        print("wrote", args.out)


def main() -> None:
    p = argparse.ArgumentParser(description="Stream text -> motion and view it (skeleton).")
    p.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    p.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    p.add_argument("--ckpt", default="checkpoints/generator/generator_transformer.pt")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int, default=49, help="max token steps (~49 = 9.8s @ 20fps)")
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fixed_length", action="store_true", help="run all --steps (no END self-stop)")
    p.add_argument("--interactive", action="store_true", help="open the aitviewer window instead of MP4")
    p.add_argument("--out", default="outputs/demo.mp4")
    run(p.parse_args())


if __name__ == "__main__":
    main()
