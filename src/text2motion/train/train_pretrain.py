from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from text2motion.model.generator import MotionGenerator, token_ce_loss
from text2motion.shared.config import load_config
from text2motion.shared.run_log import log_metrics, start_run
from text2motion.shared.seed import seed_everything


class TokenPack(Dataset):
    def __init__(self, path: str) -> None:
        self._z = np.load(path)
        self.keys = list(self._z.keys())
        if not self.keys:
            raise RuntimeError(f"empty token pack: {path}")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self._z[self.keys[i]].astype(np.int64))  # (T', R)


def collate(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    r = batch[0].shape[1]
    length_max = max(b.shape[0] for b in batch)
    out = torch.zeros(len(batch), length_max, r, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        out[i, : b.shape[0]] = b
        lengths[i] = b.shape[0]
    return out, lengths


def run(a: argparse.Namespace) -> None:
    seed_everything(2026, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(a.config)
    vocab = 1
    for lv in cfg.tokenizer.fsq_levels:
        vocab *= lv  # FSQ implicit codebook = product of levels (no tokenizer load needed)

    n_layers = cfg.generator.mamba_n_layers if a.backbone == "mamba" else cfg.generator.n_layers
    gc = replace(
        cfg.generator,
        backbone=a.backbone,
        n_layers=n_layers,
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=vocab,
    )
    generator = MotionGenerator(gc).to(dev)
    print(
        f"pretrain {a.backbone}: {sum(p.numel() for p in generator.parameters()):,} params, "
        f"codebooks {gc.num_codebooks} x vocab {vocab}"
    )

    loader = DataLoader(
        TokenPack(a.token_pack),
        batch_size=a.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=a.num_workers,
    )
    opt = torch.optim.AdamW(
        generator.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    total = a.epochs * len(loader)
    warm = cfg.train.warmup_steps
    floor = cfg.train.lr_min_ratio

    def lr_mult(step: int) -> float:
        if step < warm:
            return step / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))

    out = a.out or f"checkpoints/generator_{a.backbone}_pretrained.pt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    resume_path = Path(out).with_name(
        Path(out).stem + "_last.pt"
    )  # resume state (gen+opt+step+epoch)

    run_dir = start_run(f"pretrain_{a.backbone}", cfg, cfg.paths.outputs_dir, vars(a))
    amp_on = (
        cfg.train.amp == "bf16" and dev == "cuda"
    )  # match the fine-tune's precision (~2x faster)
    print(
        f"segments {len(loader.dataset)}  steps/epoch {len(loader)}  device {dev}  amp {cfg.train.amp}"
    )
    step = 0
    start_epoch = 0
    if a.resume and resume_path.is_file():  # continue a crashed/stalled pretrain (no progress lost)
        state = torch.load(resume_path, map_location=dev)
        generator.load_state_dict(state["generator"])
        opt.load_state_dict(state["optimizer"])
        step, start_epoch = state["step"], state["epoch"] + 1
        print(f"resumed pretrain from {resume_path} at epoch {start_epoch} (step {step})")

    for epoch in range(start_epoch, a.epochs):
        generator.train()
        tot = 0.0
        n = 0
        for tokens, lengths in loader:
            tokens, lengths = tokens.to(dev), lengths.to(dev)
            null = torch.zeros(tokens.size(0), gc.text_prefix_len, gc.d_text, device=dev)  # uncond
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=amp_on
            ):  # match the fine-tune
                loss = token_ce_loss(generator(tokens, null), tokens, lengths)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            for group in opt.param_groups:
                group["lr"] = cfg.train.lr * lr_mult(step)
            opt.step()
            step += 1
            tot += loss.item()
            n += 1
        log_metrics(run_dir, {"epoch": epoch + 1, "ce": tot / n})
        print(f"pretrain ep {epoch + 1:3d}  ce {tot / n:.4f}")
        torch.save(generator.state_dict(), out)  # the prior, kept current every epoch (crash-safe)
        torch.save(
            {
                "generator": generator.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "epoch": epoch,
            },
            resume_path,
        )
        if dev == "cuda":
            torch.cuda.empty_cache()  # defrag at epoch boundary (4GB-card fragmentation OOM lesson)

    Path(out + ".done").write_text(f"epochs {a.epochs}", encoding="utf-8")  # completion marker
    print(f"saved pretrained init -> {out}  (fine-tune with: train_generator --init_ckpt {out})")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Unconditional AMASS-token generator pretraining (E7b)."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", required=True, choices=["mamba", "transformer"])
    p.add_argument("--token_pack", default="data/amass_tokens_fsq8x1024.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true", help="continue from <out stem>_last.pt")
    run(p.parse_args())


if __name__ == "__main__":
    main()
