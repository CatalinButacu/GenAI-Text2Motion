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
from text2motion.train.trainer import GeneratorTrainer


def run(a: argparse.Namespace) -> None:
    dev = "cuda"
    free0, total = torch.cuda.mem_get_info()
    print(
        f"GPU total {total / 1e9:.2f} GB   free at start {free0 / 1e9:.2f} GB "
        f"(Windows already holds {(total - free0) / 1e9:.2f} GB)"
    )

    cfg = load_config(a.config)
    tok = ResidualFsqTokenizer(cfg.tokenizer)
    tok.load_state_dict(torch.load(a.tokenizer_ckpt, map_location="cpu"))
    tok.to(dev).eval()
    nl = cfg.generator.mamba_n_layers if a.backbone == "mamba" else cfg.generator.n_layers
    gc = replace(
        cfg.generator,
        backbone=a.backbone,
        n_layers=nl,
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=tok.codebook_size,
        use_kernel=False,
    )
    generator = MotionGenerator(gc).to(dev)
    te = CLIPTextEncoder(cfg.text_encoder).to(dev)
    out = Path(cfg.paths.hml3d_out_dir)
    mean = np.load(out / "Mean.npy").astype(np.float32)
    std = np.load(out / "Std.npy").astype(np.float32)
    trainer = GeneratorTrainer(generator, tok, cfg.train, te, mean, std)
    trainer.build_scheduler(1000)

    params = sum(p.numel() for p in generator.parameters())
    resident = torch.cuda.memory_allocated()
    print(f"{a.backbone} generator: {params / 1e6:.1f}M params")
    print(f"resident weights (gen+tokenizer+CLIP): {resident / 1e9:.2f} GB")
    print(f"  predicted params+grad+AdamW alone: {params * 16 / 1e9:.2f} GB (16 bytes/param)")

    torch.cuda.reset_peak_memory_stats()
    motion = torch.randn(a.batch, 196, 263, device=dev)
    lengths = torch.full((a.batch,), 196, dtype=torch.long, device=dev)
    caps = ["a person walks forward and sits down"] * a.batch
    try:
        text_emb = trainer.encode(caps)
        parts = trainer.train_step(motion, text_emb, lengths)
        peak = torch.cuda.max_memory_allocated()
        free_now, _ = torch.cuda.mem_get_info()
        print(
            f"\nbatch {a.batch}: PEAK {peak / 1e9:.2f} GB   (one full train_step OK, "
            f"loss {parts['total']:.3f})"
        )
        print(
            f"free remaining after step: {free_now / 1e9:.2f} GB  ->  VERDICT: FITS at batch {a.batch}"
        )
    except torch.cuda.OutOfMemoryError as exc:
        print(f"\nbatch {a.batch}: OUT OF MEMORY  ->  VERDICT: does NOT train at batch {a.batch}")
        print(f"  ({str(exc).splitlines()[0]})")


def main() -> None:
    p = argparse.ArgumentParser(description="Probe true training memory peak.")
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument("--batch", type=int, default=1)
    run(p.parse_args())


if __name__ == "__main__":
    main()
