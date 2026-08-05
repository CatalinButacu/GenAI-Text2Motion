from __future__ import annotations

import argparse
from pathlib import Path

import torch

from text2motion.render.studio_viewer import BACKBONES, ROOT, StreamingStudioViewer
from text2motion.shared.seed import seed_everything
from text2motion.stream.service import DEFAULT_HOST, DEFAULT_PORT, connect_or_spawn

DEFAULT_MODEL_DIR = str(ROOT / "data" / "smplx_models")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive streaming text->motion studio (aitviewer)."
    )
    p.add_argument("--config", default="configs/generator/final100m_fsq8x1024.yaml")
    p.add_argument("--ckpt", default="checkpoints/generator/generator_transformer_100m_val163.pt")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument("--backbone", default="transformer", choices=list(BACKBONES))
    p.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="SMPL-X model dir; if missing the studio shows the skeleton only.",
    )
    p.add_argument("--steps", type=int, default=49)
    p.add_argument("--cfg_scale", type=float, default=6.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(2026, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] connecting to the motion service (backbone={args.backbone})...")
    client = connect_or_spawn(
        args.config, args.ckpt, args.tokenizer_ckpt, args.backbone, args.host, args.port
    )
    model_dir = args.model_dir if args.model_dir and Path(args.model_dir).exists() else ""
    if not model_dir:
        print("[load] SMPL-X model dir not found -> skeleton-only studio")

    viewer = StreamingStudioViewer(client=client, model_dir=model_dir, device=device, args=args)
    viewer.run()


if __name__ == "__main__":
    main()
