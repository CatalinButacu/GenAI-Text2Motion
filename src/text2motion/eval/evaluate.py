"""THE citable generation-eval driver (promoted from the root `_twin_eval.py` scratch).

20-rep protocol on a full split (Guo et al. matcher, our_vab text, Comp_v6 stats, 32-pool):
FID, R-precision top-1/2/3, MM-Dist, Diversity — plus **MultiModality** (Guo definition:
``mm_repeats`` generations per caption on ``mm_clips`` clips, 10 random pairs each). Generation is
stream-based (the deployed path) at the GT token length unless ``--length_mode end`` (END-token
models). Every invocation writes a run manifest (CLAUDE.md logging rule). Model selection happens
on --split val; --split test is touched once per final table.

    python -m text2motion.eval.evaluate --backbone both --split test --cfg_scale 5.0 --temperature 1.1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.dataset import parse_text_file
from text2motion.eval.matcher import load_eval_stats, load_matchers
from text2motion.eval.metrics import diversity, fid, mm_dist, r_precision
from text2motion.eval.word_vectorizer import WordVectorizer
from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import load_config
from text2motion.shared.run_log import log_metrics, start_run


def make_build_text(w_vec):
    def build_text(tokens):
        items = ["sos/OTHER"] + tokens[:20] + ["eos/OTHER"]
        word_embs = np.stack([w_vec[item][0] for item in items]).astype(np.float32)
        pos_onehots = np.stack([w_vec[item][1] for item in items]).astype(np.float32)
        return word_embs, pos_onehots

    return build_text


def embed_motions(matcher, feats, mean, std, device, batch=32):
    feats = [f[: (f.shape[0] // 4) * 4] for f in feats]
    out = []
    for start in range(0, len(feats), batch):
        group = feats[start : start + batch]
        max_t = max(f.shape[0] for f in group)
        x = np.zeros((len(group), max_t, 263), np.float32)
        lengths = [f.shape[0] for f in group]
        for row, feat in enumerate(group):
            x[row, : feat.shape[0]] = (feat - mean) / std
        with torch.no_grad():
            emb = matcher(torch.from_numpy(x).to(device), torch.tensor(lengths, device=device))
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


def embed_texts(matcher, pairs, device, batch=32):
    out = []
    for start in range(0, len(pairs), batch):
        group = pairs[start : start + batch]
        max_l = max(we.shape[0] for we, _ in group)
        we_pad = np.zeros((len(group), max_l, 300), np.float32)
        pe_pad = np.zeros((len(group), max_l, 15), np.float32)
        lengths = [we.shape[0] for we, _ in group]
        for row, (we, pe) in enumerate(group):
            we_pad[row, : we.shape[0]] = we
            pe_pad[row, : pe.shape[0]] = pe
        with torch.no_grad():
            emb = matcher(
                torch.from_numpy(we_pad).to(device),
                torch.from_numpy(pe_pad).to(device),
                lengths=torch.tensor(lengths, device=device),
            )
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


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


def stream_motion(generator, tokenizer, text_emb, token_len, args, our_mean, our_std):
    """One stream->decode pass; text_emb (B, ...) -> list of B raw (T, 263) features."""
    steps = list(
        generator.stream(
            text_emb,
            token_len,
            temperature=args.temperature,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
            stop_at_end=args.length_mode == "end",
        )
    )
    if not steps:  # END on the very first step: no usable motion (counted by the caller)
        return None
    tokens = torch.stack(steps, dim=1)
    decoded = tokenizer.decode(tokens).cpu().numpy() * our_std + our_mean
    return [decoded[row] for row in range(decoded.shape[0])]


@torch.no_grad()
def generate_split(
    generator, tokenizer, text_encoder, cfg, ids, out_dir, text_dir, our_mean, our_std, args
):
    gt_feats, gen_feats, tok_lists, captions = [], [], [], []
    for clip_id in ids[: args.max_clips]:
        vec_path = out_dir / "new_joint_vecs" / f"{clip_id}.npy"
        text_path = text_dir / f"{clip_id}.txt"
        if not vec_path.is_file() or not text_path.is_file():
            continue
        feat = np.load(vec_path).astype(np.float32)
        if np.isnan(feat).any():
            continue
        token_len = min(feat.shape[0], 196) // cfg.tokenizer.downsample
        if token_len < 2:
            continue
        feat = feat[: token_len * cfg.tokenizer.downsample]
        annotations = parse_text_file(text_path)
        token_lists = [a.tokens for a in annotations if a.tokens]
        if not token_lists:
            continue
        text_emb = text_encoder([annotations[0].caption])  # condition on the first caption
        decoded = stream_motion(generator, tokenizer, text_emb, token_len, args, our_mean, our_std)
        if decoded is None:
            continue
        gt_feats.append(feat)
        gen_feats.append(decoded[0])
        tok_lists.append(token_lists)
        captions.append(annotations[0].caption)
    return gt_feats, gen_feats, tok_lists, captions


@torch.no_grad()
def multimodality(
    generator,
    tokenizer,
    text_encoder,
    captions,
    token_lens,
    motion_matcher,
    eval_mean,
    eval_std,
    our_mean,
    our_std,
    device,
    args,
):
    """Guo MultiModality: per caption, mm_repeats generations -> 10 random embedding pairs ->
    mean pairwise distance, averaged over mm_clips captions."""
    rng = np.random.default_rng(0)
    per_caption = []
    for caption, token_len in zip(captions[: args.mm_clips], token_lens[: args.mm_clips]):
        text_emb = text_encoder([caption] * args.mm_repeats)
        decoded = stream_motion(generator, tokenizer, text_emb, token_len, args, our_mean, our_std)
        if decoded is None or len(decoded) < 2:
            continue
        emb = embed_motions(motion_matcher, decoded, eval_mean, eval_std, device)
        first = rng.integers(0, len(emb), 10)
        second = rng.integers(0, len(emb), 10)
        per_caption.append(float(np.linalg.norm(emb[first] - emb[second], axis=1).mean()))
    return float(np.mean(per_caption)) if per_caption else float("nan")


def score(
    name,
    gt_feats,
    gen_feats,
    tok_lists,
    motion_matcher,
    text_matcher,
    build_text,
    eval_mean,
    eval_std,
    device,
    reps=20,
):
    gt_emb = embed_motions(motion_matcher, gt_feats, eval_mean, eval_std, device)
    gen_emb = embed_motions(motion_matcher, gen_feats, eval_mean, eval_std, device)

    flat, owner = [], []
    for clip_index, token_lists in enumerate(tok_lists):
        for tokens in token_lists:
            flat.append(build_text(tokens))
            owner.append(clip_index)
    text_emb = embed_texts(text_matcher, flat, device)
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
    parser.add_argument("--ckpt", default=None, help="explicit checkpoint (single-backbone only)")
    parser.add_argument(
        "--tokenizer_ckpt",
        default="checkpoints/tokenizer_fsq.pt",
        help="frozen tokenizer state_dict to load; must match cfg.tokenizer architecture",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.paths.hml3d_out_dir)
    text_dir = Path(cfg.paths.texts_dir) if cfg.paths.texts_dir else out_dir / "texts"
    our_mean = np.load(out_dir / "Mean.npy").astype(np.float32)
    our_std = np.load(out_dir / "Std.npy").astype(np.float32)
    eval_mean, eval_std = load_eval_stats(cfg.paths.eval_stats_dir)
    motion_matcher, text_matcher = load_matchers(cfg.paths.eval_matcher, device=device)
    build_text = make_build_text(WordVectorizer(r"data/t2m_glove/glove", "our_vab"))

    tokenizer = ResidualFsqTokenizer(cfg.tokenizer)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location="cpu"))
    tokenizer.to(device).eval()
    text_encoder = CLIPTextEncoder(cfg.text_encoder).to(device).eval()

    ids = [n.strip() for n in (out_dir / f"{args.split}.txt").read_text().splitlines() if n.strip()]
    ids = list(dict.fromkeys(i[1:] if i.startswith("M") else i for i in ids))

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
        gt_feats, gen_feats, tok_lists, captions = generate_split(
            generator,
            tokenizer,
            text_encoder,
            cfg,
            ids,
            out_dir,
            text_dir,
            our_mean,
            our_std,
            args,
        )
        metrics = score(
            backbone,
            gt_feats,
            gen_feats,
            tok_lists,
            motion_matcher,
            text_matcher,
            build_text,
            eval_mean,
            eval_std,
            device,
        )
        if args.mm_clips > 0:
            token_lens = [min(f.shape[0], 196) // cfg.tokenizer.downsample for f in gt_feats]
            metrics["multimodality"] = multimodality(
                generator,
                tokenizer,
                text_encoder,
                captions,
                token_lens,
                motion_matcher,
                eval_mean,
                eval_std,
                our_mean,
                our_std,
                device,
                args,
            )
        log_metrics(run_dir, {"backbone": backbone, "ckpt": ckpt, **metrics})
        print(
            f"{backbone:12s} clips {metrics['clips']:4d}  FID {metrics['fid']:6.3f}  "
            f"R@1 {metrics['r_top1']:.3f}±{metrics['r_top1_std']:.3f}  "
            f"R@2 {metrics['r_top2']:.3f}  R@3 {metrics['r_top3']:.3f}  "
            f"MM {metrics['mm_dist']:.3f}  Div {metrics['diversity']:.3f}  "
            f"MModality {metrics.get('multimodality', float('nan')):.3f}"
        )
        del generator
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
