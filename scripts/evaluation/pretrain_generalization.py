# python scripts/evaluation/pretrain_generalization.py --config configs/generator/gen_pilot_fsq8x1024.yaml --ckpt artifacts/checkpoints/generator/pretrain.pt
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.run_log import log_metrics, start_run
from text2motion.generation.contracts import Backbone
from text2motion.generation.nn_model import token_ce_loss
from text2motion.motion.storage import sha256_file
from text2motion.tokenization.corpus import pack_clip_keys, read_pack_identity
from text2motion.tokenization.nn_model import TokenizerNetwork


def seen_clip_stems(token_pack: str) -> set[str]:
    with np.load(token_pack) as pack:
        return {key.rsplit("__", 1)[0] for key in pack_clip_keys(pack)}


def split_clips(features_dir: str, seen: set[str], min_frames: int) -> tuple[list, list]:
    held_out = []
    trained = []
    for path in sorted(Path(features_dir).glob("*.npy")):
        length = int(np.load(path, mmap_mode="r").shape[0])
        if length < min_frames:
            continue
        if path.stem in seen:
            trained.append((path, length))
        else:
            held_out.append((path, length))
    return held_out, trained


@torch.no_grad()
def encode_clip(
    tokenizer: TokenizerNetwork,
    path: Path,
    frames: int,
    mean,
    std,
    device,
    downsample: int,
) -> torch.Tensor:
    feats = np.load(path).astype(np.float32)[:frames]
    usable = (feats.shape[0] // downsample) * downsample
    normalized = (feats[:usable] - mean) / std
    batch = torch.from_numpy(normalized).unsqueeze(0).to(device)
    return tokenizer.encode(batch)[0].cpu()


@torch.no_grad()
def mean_ce(generator, gen_cfg, token_seqs: list[torch.Tensor], device, batch_size: int) -> float:
    total = 0.0
    counted = 0
    for start in range(0, len(token_seqs), batch_size):
        group = token_seqs[start : start + batch_size]
        longest = max(seq.shape[0] for seq in group)
        tokens = torch.zeros(len(group), longest, group[0].shape[1], dtype=torch.long)
        lengths = torch.zeros(len(group), dtype=torch.long)
        for row, seq in enumerate(group):
            tokens[row, : seq.shape[0]] = seq
            lengths[row] = seq.shape[0]
        tokens = tokens.to(device)
        lengths = lengths.to(device)
        null = torch.zeros(tokens.size(0), gen_cfg.text_prefix_len, gen_cfg.d_text, device=device)
        loss = token_ce_loss(generator(tokens, null), tokens, lengths)
        total += float(loss) * len(group)
        counted += len(group)
    return total / counted


def run(args: argparse.Namespace) -> None:
    app = ApplicationBootstrap.from_config(args.config, device=args.device, seed=2026)
    cfg = app.config
    device = app.device
    rng = np.random.default_rng(2026)

    pack_tokenizer = read_pack_identity(args.token_pack)["tokenizer_sha256"]
    if pack_tokenizer != sha256_file(args.tokenizer_ckpt):
        raise ValueError(
            f"{args.token_pack} was built with another tokenizer than {args.tokenizer_ckpt}"
        )
    tokenizer = app.load_tokenizer(args.tokenizer_ckpt).network
    spec = app.resolve_generator_spec(Backbone(args.backbone), tokenizer.codebook_size)
    gen_cfg = spec.config
    generator = spec.create_model(device)
    generator.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    generator.eval()
    params = sum(p.numel() for p in generator.parameters())
    print(f"{args.ckpt}: {params / 1e6:.2f}M params, prefix {gen_cfg.text_prefix_len}")

    seen = seen_clip_stems(args.token_pack)
    held_out, trained = split_clips(args.features_dir, seen, args.min_frames)
    print(f"clips: {len(held_out)} never-trained, {len(trained)} trained")
    if len(held_out) < args.num_clips or len(trained) < args.num_clips:
        raise RuntimeError("not enough clips in one of the two pools for the requested sample")

    held_idx = rng.choice(len(held_out), args.num_clips, replace=False)
    held_sample = [held_out[i] for i in held_idx]
    lengths = [length for _, length in held_sample]

    trained_idx = rng.choice(len(trained), args.num_clips, replace=False)
    trained_sample = [(trained[i][0], lengths[k]) for k, i in enumerate(trained_idx)]

    scaler = app.create_motion_repository().load_scaler()
    mean, std = scaler.mean, scaler.std

    run_dir = start_run("pretrain_generalization", cfg, cfg.paths.logs_dir, vars(args))

    downsample = cfg.tokenizer.downsample
    held_tokens = [
        encode_clip(tokenizer, p, n, mean, std, device, downsample) for p, n in held_sample
    ]
    trained_tokens = [
        encode_clip(tokenizer, p, n, mean, std, device, downsample) for p, n in trained_sample
    ]

    held_ce = mean_ce(generator, gen_cfg, held_tokens, device, args.batch_size)
    trained_ce = mean_ce(generator, gen_cfg, trained_tokens, device, args.batch_size)
    gap = held_ce - trained_ce

    print(f"trained-clip CE   {trained_ce:.4f}   (perplexity {np.exp(trained_ce):.2f})")
    print(f"held-out clip CE  {held_ce:.4f}   (perplexity {np.exp(held_ce):.2f})")
    print(f"generalization gap {gap:+.4f}  ratio {held_ce / trained_ce:.3f}x")

    log_metrics(
        run_dir,
        {
            "ckpt": args.ckpt,
            "backbone": args.backbone,
            "params": params,
            "clips_per_pool": args.num_clips,
            "length_matched": True,
            "mean_frames": float(np.mean(lengths)),
            "trained_ce": trained_ce,
            "held_out_ce": held_ce,
            "gap": gap,
            "trained_perplexity": float(np.exp(trained_ce)),
            "held_out_perplexity": float(np.exp(held_ce)),
            "pool_held_out": len(held_out),
            "pool_trained": len(trained),
        },
    )
    print(f"run dir: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--backbone", default="transformer", choices=["transformer", "mamba"])
    parser.add_argument(
        "--tokenizer_ckpt", default="artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
    )
    parser.add_argument("--token_pack", default="data/amass_tokens_fsq8x1024.npz")
    parser.add_argument("--features_dir", default="data/AMASS_263/new_joint_vecs")
    parser.add_argument("--num_clips", type=int, default=300)
    parser.add_argument("--min_frames", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
