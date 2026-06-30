"""Encode the pretraining corpus through the FROZEN tokenizer -> one compact token pack (E7b).

Each 263-feature sequence is normalized with the OFFICIAL HumanML3D Mean/Std (the tokenizer's
training space — corpus-specific stats would shift the lattice), segmented into ``segment_frames``
windows (default 196 = the generator's horizon; stride configurable for overlap), and encoded to
``(T', num_quantizers)`` int16 indices. Output: a single compressed ``.npz`` (segment name -> token
array) — the entire AMASS corpus collapses to tens of MB, which is what we ship to S3 for cloud
pretraining instead of 151 GB of mocap.

    python -m text2motion.data.hml3d.tokenize_corpus --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import load_config
from text2motion.shared.run_log import log_metrics, start_run


@torch.no_grad()
def tokenize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg.paths.hml3d_out_dir)
    mean = np.load(out_dir / "Mean.npy").astype(np.float32)
    std = np.load(out_dir / "Std.npy").astype(np.float32)

    tokenizer = ResidualFsqTokenizer(cfg.tokenizer)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location="cpu"))
    tokenizer.to(device).eval()
    downsample = cfg.tokenizer.downsample

    features = sorted(Path(args.features_dir).glob("*.npy"))
    if not features:
        raise FileNotFoundError(
            f"no features under {args.features_dir} — run pretrain_corpus first"
        )

    run_dir = start_run("tokenize_corpus", cfg, cfg.paths.outputs_dir, vars(args))
    pack: dict[str, np.ndarray] = {}
    segments = tokens_total = 0
    for path in tqdm(features, desc="encode"):
        feat = (np.load(path).astype(np.float32) - mean) / std
        for start in range(0, feat.shape[0] - args.segment_frames + 1, args.stride):
            window = feat[start : start + args.segment_frames]
            usable = (window.shape[0] // downsample) * downsample
            batch = torch.from_numpy(window[:usable]).unsqueeze(0).to(device)
            indices = tokenizer.encode(batch)[0].cpu().numpy().astype(np.int16)
            pack[f"{path.stem}__{start}"] = indices
            segments += 1
            tokens_total += indices.shape[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **pack)
    summary = {
        "sequences": len(features),
        "segments": segments,
        "token_steps": tokens_total,
        "pack_mb": round(out_path.stat().st_size / 1e6, 1),
    }
    log_metrics(run_dir, summary)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize the pretraining corpus (E7b pack).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--features_dir", default="data/AMASS_263/new_joint_vecs")
    parser.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/tokenizer_fsq.pt")
    parser.add_argument("--out", default="data/amass_tokens.npz")
    parser.add_argument("--segment_frames", type=int, default=196)
    parser.add_argument("--stride", type=int, default=196, help="< segment_frames for overlap")
    args = parser.parse_args()
    tokenize(args)


if __name__ == "__main__":
    main()
