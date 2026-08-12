from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.dataset import parse_text_file
from text2motion.eval.context import EvalContext, GenerationPipeline, SamplingCfg
from text2motion.eval.embedding import embed_motions, embed_texts
from text2motion.eval.metrics import bootstrap_fid, diversity, fid, mm_dist, r_precision
from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import load_config
from text2motion.shared.run_log import log_metrics, start_run
from text2motion.shared.seed import seed_everything


def load_generator(backbone, cfg, tokenizer, device, ckpt=None):
    n_layers = cfg.generator.mamba_n_layers if backbone == "mamba" else cfg.generator.n_layers
    gen_cfg = replace(
        cfg.generator,
        backbone=backbone,
        n_layers=n_layers,
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=tokenizer.codebook_size,
    )
    generator = MotionGenerator(gen_cfg).to(device).eval()
    ckpt = ckpt or f"checkpoints/generator_{backbone}.pt"
    generator.load_state_dict(torch.load(ckpt, map_location=device))
    return generator


@torch.no_grad()
def generate_split(pipeline, ctx, sampling, ids, max_clips):
    gt_feats, gen_feats, tok_lists, captions = [], [], [], []
    requested = list(ids[:max_clips])
    for clip_id in requested:
        vec_path = ctx.out_dir / "new_joint_vecs" / f"{clip_id}.npy"
        text_path = ctx.text_dir / f"{clip_id}.txt"
        if not vec_path.is_file() or not text_path.is_file():
            continue
        feat = np.load(vec_path).astype(np.float32)
        if np.isnan(feat).any():
            continue
        token_len = min(feat.shape[0], 196) // ctx.downsample
        if token_len < 2:
            continue
        feat = feat[: token_len * ctx.downsample]
        annotations = parse_text_file(text_path)
        token_lists = [a.tokens for a in annotations if a.tokens]
        if not token_lists:
            continue
        caption = annotations[0].caption  # condition on the first caption
        decoded = pipeline.generate_batch([caption], token_len, sampling, ctx)
        if decoded is None:
            continue
        gt_feats.append(feat)
        gen_feats.append(decoded[0])
        tok_lists.append(token_lists)
        captions.append(caption)

    if len(gt_feats) != len(requested):
        print(
            f"WARNING: scored {len(gt_feats)} of {len(requested)} requested clips "
            f"({len(requested) - len(gt_feats)} dropped: missing file, non-finite, too short, "
            f"no caption tokens, or empty generation). FID is only comparable at equal clip counts."
        )
    return gt_feats, gen_feats, tok_lists, captions


@torch.no_grad()
def multimodality(pipeline, ctx, sampling, captions, token_lens, mm_clips, mm_repeats):
    rng = np.random.default_rng(0)
    per_caption = []
    for caption, token_len in zip(captions[:mm_clips], token_lens[:mm_clips]):
        decoded = pipeline.generate_batch([caption] * mm_repeats, token_len, sampling, ctx)
        if decoded is None or len(decoded) < 2:
            continue
        emb = embed_motions(ctx.motion_matcher, decoded, ctx.eval_mean, ctx.eval_std, ctx.device)
        first = rng.integers(0, len(emb), 10)
        second = rng.integers(0, len(emb), 10)
        per_caption.append(float(np.linalg.norm(emb[first] - emb[second], axis=1).mean()))
    return float(np.mean(per_caption)) if per_caption else float("nan")


def score(name, gt_feats, gen_feats, tok_lists, ctx, reps=20, bootstrap=200):
    gt_emb = embed_motions(ctx.motion_matcher, gt_feats, ctx.eval_mean, ctx.eval_std, ctx.device)
    gen_emb = embed_motions(ctx.motion_matcher, gen_feats, ctx.eval_mean, ctx.eval_std, ctx.device)

    flat, owner = [], []
    for clip_index, token_lists in enumerate(tok_lists):
        for tokens in token_lists:
            flat.append(ctx.build_text(tokens))
            owner.append(clip_index)
    text_emb = embed_texts(ctx.text_matcher, flat, ctx.device)
    owner = np.array(owner)
    per_clip = [np.where(owner == c)[0] for c in range(len(gen_feats))]

    rng = np.random.default_rng(0)
    rprec, mmdist = [], []
    for _ in range(reps):
        chosen = np.array([rng.choice(indices) for indices in per_clip])
        sel = text_emb[chosen]
        perm = rng.permutation(len(gen_emb))
        rprec.append(r_precision(sel[perm], gen_emb[perm], pool_size=32, top_k=3))
        mmdist.append(mm_dist(sel[perm], gen_emb[perm]))
    rprec = np.stack(rprec)
    return {
        "clips": len(gen_feats),
        "fid": fid(gt_emb, gen_emb),
        **bootstrap_fid(gt_emb, gen_emb, resamples=bootstrap),
        "r_top1": float(rprec[:, 0].mean()),
        "r_top1_std": float(rprec[:, 0].std()),
        "r_top2": float(rprec[:, 1].mean()),
        "r_top3": float(rprec[:, 2].mean()),
        "mm_dist": float(np.mean(mmdist)),
        "diversity": diversity(gen_emb, num_pairs=300),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Citable generation eval (20-rep protocol).")
    parser.add_argument("--backbone", default="both", choices=["mamba", "transformer", "both"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max_clips", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--length_mode", default="fixed", choices=["fixed", "end"])
    parser.add_argument("--mm_clips", type=int, default=100, help="0 disables MultiModality")
    parser.add_argument("--mm_repeats", type=int, default=30)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="clip-level bootstrap resamples for the FID interval (0 disables)",
    )
    parser.add_argument("--ckpt", default=None, help="explicit checkpoint (single-backbone only)")
    parser.add_argument(
        "--tokenizer_ckpt",
        default="checkpoints/tokenizer/tokenizer_fsq.pt",
        help="frozen tokenizer state_dict to load; must match cfg.tokenizer architecture",
    )
    parser.add_argument(
        "--text_encoder_ckpt",
        default=None,
        help="trainer <name>_last.pt holding the co-adapted CLIP; omit only for a fully frozen encoder",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="overrides cfg.seed; fixes the sampling RNG"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    seed_everything(
        seed, cfg.deterministic
    )  # sampling is torch-RNG driven: unseeded eval is NOT reproducible
    ctx = EvalContext.from_config(cfg, device=args.device)
    sampling = SamplingCfg.from_args(args)
    device = ctx.device

    tokenizer = ResidualFsqTokenizer(cfg.tokenizer)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location="cpu"))
    tokenizer.to(device).eval()
    text_encoder = CLIPTextEncoder(cfg.text_encoder).to(device).eval()
    encoder_ckpt = args.text_encoder_ckpt
    if encoder_ckpt is None and args.ckpt:
        sidecar = Path(args.ckpt).with_name(Path(args.ckpt).stem + "_text_encoder.pt")
        if sidecar.is_file():
            encoder_ckpt = str(sidecar)
            print(f"auto-discovered co-adapted text encoder: {sidecar.name}")
    if encoder_ckpt:
        args.text_encoder_ckpt = encoder_ckpt
        state = torch.load(encoder_ckpt, map_location=device, weights_only=False)
        if "text_encoder" not in state:
            raise KeyError(
                f"{args.text_encoder_ckpt} has no 'text_encoder' entry (keys: {list(state)}); "
                f"pass the trainer's <name>_last.pt, which stores the co-adapted encoder"
            )
        text_encoder.load_state_dict(state["text_encoder"])
        text_encoder.to(device).eval()
        print(f"loaded co-adapted text encoder from {args.text_encoder_ckpt}")
    elif cfg.text_encoder.unfreeze_last_n > 0 or cfg.text_encoder.unfreeze_projection:
        print(
            "WARNING: config unfreezes the text encoder during training but no --text_encoder_ckpt "
            "was given, so generation runs against PRISTINE CLIP -- not the encoder this generator "
            "was trained with. Numbers are not comparable to in-training eval."
        )

    ids = ctx.clip_ids(args.split)

    run_dir = start_run(f"evaluate_{args.split}", cfg, cfg.paths.outputs_dir, vars(args))
    backbones = ["transformer", "mamba"] if args.backbone == "both" else [args.backbone]
    print(
        f"device {device}  {args.split} ids {len(ids)}  cfg_scale {args.cfg_scale}  "
        f"temp {args.temperature}  top_p {args.top_p}  length {args.length_mode}  (20-rep, our_vab)"
    )
    for backbone in backbones:
        ckpt = args.ckpt if len(backbones) == 1 else None
        if not Path(ckpt or f"checkpoints/generator_{backbone}.pt").is_file():
            print(f"{backbone:12s} SKIP (no checkpoint yet)")
            continue
        generator = load_generator(backbone, cfg, tokenizer, device, ckpt)
        pipeline = GenerationPipeline(generator, text_encoder, tokenizer).eval()
        gt_feats, gen_feats, tok_lists, captions = generate_split(
            pipeline, ctx, sampling, ids, args.max_clips
        )
        metrics = score(backbone, gt_feats, gen_feats, tok_lists, ctx, bootstrap=args.bootstrap)
        if args.mm_clips > 0:
            token_lens = [min(f.shape[0], 196) // ctx.downsample for f in gt_feats]
            metrics["multimodality"] = multimodality(
                pipeline, ctx, sampling, captions, token_lens, args.mm_clips, args.mm_repeats
            )
        log_metrics(run_dir, {"backbone": backbone, "ckpt": ckpt, **metrics})
        print(
            f"{backbone:12s} clips {metrics['clips']:4d}  FID {metrics['fid']:6.3f}  "
            f"R@1 {metrics['r_top1']:.3f}+/-{metrics['r_top1_std']:.3f}  "
            f"R@2 {metrics['r_top2']:.3f}  R@3 {metrics['r_top3']:.3f}  "
            f"MM {metrics['mm_dist']:.3f}  Div {metrics['diversity']:.3f}  "
            f"MModality {metrics.get('multimodality', float('nan')):.3f}"
        )
        del generator
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
