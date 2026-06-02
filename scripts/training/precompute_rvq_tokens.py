#!/usr/bin/env python
"""Precompute RVQ token cache for SSM training.

Rationale
---------
SSM training runs ``frozen_tokenizer.encode(motion)`` once per batch per epoch.
With the RVQ tokenizer frozen, this is pure waste: the (motion -> tokens) map
is deterministic. Cache it once. The trainer then reads ``(text, tokens)`` and
spends 100% of its compute on the SSM itself, not on the encode-step.

What the cache contains
-----------------------
A joblib pickle with:
    {
        "meta": {
            "rvq_ckpt_hash": short sha256 of the tokenizer checkpoint,
            "down_t": int,
            "n_codebooks": int,
            "codebook_size": int,
            "max_motion_length": int,
            "data_source": str,
            "data_dir": str,
        },
        "vocab": dict from the underlying MotionDataset/HumanML3DMotionDataset,
        "motion_stats": MotionStats used for normalisation,
        "samples": list[{text, tokens, length, lat_length, source?}],
    }

The trainer side loads it via ``src.data.token_dataset.TokenDataset``.

Usage
-----
    python scripts/training/precompute_rvq_tokens.py \\
        --rvq-checkpoint checkpoints/rvq_tokenizer/best_model.pt \\
        --data-source amass \\
        --data-dir data/AMASS \\
        --output data/.cache/tokens_amass.joblib
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.architecture.rvq_tokenizer import MotionRVQTokenizer
from src.data.humanml3d_loader import HumanML3DMotionDataset
from src.data.motion_dataset import MotionDataset
from src.shared.constants import SMPLX

log = logging.getLogger(__name__)


def short_hash(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()[:12]


def build_dataset(source: str, data_dir: str, max_motion_length: int,
                  max_samples: int | None, amass_dir: str) -> tuple[object, str]:
    """Match the train-side dataset construction so the cache stays aligned."""

    if source == "amass":
        ds = MotionDataset(
            data_dir, "all" if False else "train",
            max_motion_length, augment=False, max_samples=max_samples,
        )

        return ds, "amass"

    if source == "humanml3d":
        ds = HumanML3DMotionDataset(
            data_dir=data_dir, amass_dir=amass_dir,
            split="train", max_motion_length=max_motion_length,
            augment=False,
        )

        return ds, "humanml3d"

    raise ValueError(f"unsupported --data-source {source!r}")


@torch.no_grad()
def encode_split(
    tokenizer: MotionRVQTokenizer,
    dataset,
    batch_size: int,
    device: torch.device,
    down_t: int,
) -> list[dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    out: list[dict] = []

    for batch in loader:
        motion = batch["motion"].to(device)   # (B, T, 168) normalised
        mask = batch["motion_mask"].to(device)  # (B, T)
        # encode() returns indices (B, T_lat, K) on the frozen codebooks.
        tokens = tokenizer.encode(motion).cpu().numpy().astype(np.int64)
        lengths = mask.sum(dim=1).cpu().numpy().astype(np.int64)

        for i, text in enumerate(batch["texts"]):
            length = int(lengths[i])
            lat_length = max(1, (length + down_t - 1) // down_t)
            out.append({
                "text": text,
                "tokens": tokens[i, :lat_length].copy(),
                "length": length,
                "lat_length": lat_length,
            })

        if len(out) % 1000 < batch_size:
            log.info("[precompute] %d samples encoded", len(out))

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rvq-checkpoint", required=True, dest="rvq_checkpoint",
                   help="Path to the frozen RVQ tokenizer checkpoint .pt")
    p.add_argument("--data-source", choices=["amass", "humanml3d"], default="amass",
                   dest="data_source")
    p.add_argument("--data-dir", default="data/AMASS", dest="data_dir")
    p.add_argument("--amass-dir", default="data/AMASS", dest="amass_dir",
                   help="AMASS backing store (only used for --data-source humanml3d)")
    p.add_argument("--max-motion-length", type=int, default=200, dest="max_motion_length")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination joblib file, e.g. data/.cache/tokens_amass.joblib")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ckpt_path = Path(args.rvq_checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"RVQ checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # NOSONAR
    cfg = ck["config"]
    latent_dim = int(getattr(cfg, "latent_dim", cfg["latent_dim"] if isinstance(cfg, dict) else 128))
    n_codebooks = int(
        getattr(cfg, "n_codebooks", cfg["n_codebooks"] if isinstance(cfg, dict) else 6)
    )
    codebook_size = int(
        getattr(cfg, "codebook_size", cfg["codebook_size"] if isinstance(cfg, dict) else 512)
    )
    down_t = int(getattr(cfg, "down_t", cfg["down_t"] if isinstance(cfg, dict) else 4))

    log.info(
        "[precompute] tokenizer: latent_dim=%d K=%d V=%d down_t=%d",
        latent_dim, n_codebooks, codebook_size, down_t,
    )

    tokenizer = MotionRVQTokenizer(
        motion_dim=SMPLX.pose_dim,
        latent_dim=latent_dim,
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        down_t=down_t,
    ).to(device)
    tokenizer.load_state_dict(ck["model_state_dict"])
    tokenizer.eval()
    log.info("[precompute] tokenizer loaded from %s", ckpt_path)

    dataset, source = build_dataset(
        args.data_source, args.data_dir, args.max_motion_length,
        args.max_samples, args.amass_dir,
    )
    log.info("[precompute] dataset size: %d samples", len(dataset))

    samples = encode_split(tokenizer, dataset, args.batch_size, device, down_t)

    payload = {
        "meta": {
            "rvq_ckpt_hash": short_hash(ckpt_path),
            "down_t": down_t,
            "n_codebooks": n_codebooks,
            "codebook_size": codebook_size,
            "max_motion_length": args.max_motion_length,
            "data_source": source,
            "data_dir": args.data_dir,
        },
        "vocab": getattr(dataset, "vocab", {}),
        "motion_stats": getattr(dataset, "motion_stats", None),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output, compress=3)
    log.info("[precompute] cached %d samples -> %s", len(samples), args.output)


if __name__ == "__main__":
    main()
