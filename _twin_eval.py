"""20-rep full-test generation eval for the Contribution-B twin (mamba vs transformer).

Mirrors the validated GT driver `_l2.py` protocol (our_vab text, Comp_v6 motion stats, 32-pool,
20 reps) but the motion side is GENERATED (text->CLIP->stream->frozen-FSQ decode), per backbone's
best-FID checkpoint. Generation is single-sample at the GT token length; the 20 reps vary the random
caption choice + 32-pool permutation (no MultiModality). Run after both runs finish:

    $env:PYTHONPATH="src"; .venv/Scripts/python.exe _twin_eval.py --backbone both
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
from text2motion.eval.word_vectorizer import WordVectorizer  # vendored Guo our_vab (portable)
from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import load_config


def make_build_text(w_vec):
    def build_text(tokens):
        items = ["sos/OTHER"] + tokens[:20] + ["eos/OTHER"]
        we = np.stack([w_vec[i][0] for i in items]).astype(np.float32)
        pe = np.stack([w_vec[i][1] for i in items]).astype(np.float32)
        return we, pe

    return build_text


def embed_motions(matcher, feats, mean, std, device, batch=32):
    feats = [f[: (f.shape[0] // 4) * 4] for f in feats]
    out = []
    for i in range(0, len(feats), batch):
        group = feats[i : i + batch]
        max_t = max(f.shape[0] for f in group)
        x = np.zeros((len(group), max_t, 263), np.float32)
        lengths = [f.shape[0] for f in group]
        for j, f in enumerate(group):
            x[j, : f.shape[0]] = (f - mean) / std
        with torch.no_grad():
            e = matcher(torch.from_numpy(x).to(device), torch.tensor(lengths, device=device))
        out.append(e.cpu().numpy())
    return np.concatenate(out)


def embed_texts(matcher, pairs, device, batch=32):
    out = []
    for i in range(0, len(pairs), batch):
        group = pairs[i : i + batch]
        max_l = max(we.shape[0] for we, _ in group)
        we_pad = np.zeros((len(group), max_l, 300), np.float32)
        pe_pad = np.zeros((len(group), max_l, 15), np.float32)
        lengths = [we.shape[0] for we, _ in group]
        for j, (we, pe) in enumerate(group):
            we_pad[j, : we.shape[0]] = we
            pe_pad[j, : pe.shape[0]] = pe
        with torch.no_grad():
            e = matcher(
                torch.from_numpy(we_pad).to(device),
                torch.from_numpy(pe_pad).to(device),
                lengths=torch.tensor(lengths, device=device),
            )
        out.append(e.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def generate(
    backbone,
    cfg,
    tokenizer,
    te,
    device,
    ids,
    out_dir,
    text_dir,
    our_mean,
    our_std,
    max_clips,
    temperature,
    top_p,
    cfg_scale=1.0,
    ckpt=None,
):
    n_layers = cfg.generator.mamba_n_layers if backbone == "mamba" else cfg.generator.n_layers
    gen_cfg = replace(
        cfg.generator,
        backbone=backbone,
        n_layers=n_layers,
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=tokenizer.codebook_size,
    )
    gen = MotionGenerator(gen_cfg).to(device).eval()
    ckpt = ckpt or f"checkpoints/generator_{backbone}.pt"
    gen.load_state_dict(torch.load(ckpt, map_location=device))

    gt_feats, gen_feats, tok_lists = [], [], []
    for clip_id in ids[:max_clips]:
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
        anns = parse_text_file(text_path)
        token_lists = [a.tokens for a in anns if a.tokens]
        if not token_lists:
            continue
        text_emb = te([anns[0].caption])  # condition on the first caption
        tokens = torch.stack(
            list(
                gen.stream(
                    text_emb, token_len, temperature=temperature, top_p=top_p, cfg_scale=cfg_scale
                )
            ),
            dim=1,
        )
        decoded = tokenizer.decode(tokens)[0].cpu().numpy() * our_std + our_mean
        gt_feats.append(feat)
        gen_feats.append(decoded)
        tok_lists.append(token_lists)
    return gt_feats, gen_feats, tok_lists


def evaluate(
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
        for toks in token_lists:
            flat.append(build_text(toks))
            owner.append(clip_index)
    text_emb = embed_texts(text_matcher, flat, device)
    owner = np.array(owner)
    per_clip = [np.where(owner == c)[0] for c in range(len(gen_feats))]

    rng = np.random.default_rng(0)
    rprec, mmdist = [], []
    for _ in range(reps):
        chosen = np.array([rng.choice(idxs) for idxs in per_clip])
        sel = text_emb[chosen]
        perm = rng.permutation(len(gen_emb))
        rprec.append(r_precision(sel[perm], gen_emb[perm], pool_size=32, top_k=3))
        mmdist.append(mm_dist(sel[perm], gen_emb[perm]))
    rprec = np.stack(rprec)
    fid_val = fid(gt_emb, gen_emb)
    div = diversity(gen_emb, num_pairs=300)
    print(
        f"{name:12s} clips {len(gen_feats):4d}  FID {fid_val:6.3f}  "
        f"R@1 {rprec[:, 0].mean():.3f}±{rprec[:, 0].std():.3f}  R@2 {rprec[:, 1].mean():.3f}  "
        f"R@3 {rprec[:, 2].mean():.3f}  MM {np.mean(mmdist):.3f}  Div {div:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="both", choices=["mamba", "transformer", "both"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_clips", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--split", default="test", choices=["val", "test"]
    )  # val = sweeps/selection
    parser.add_argument("--ckpt", default=None, help="explicit checkpoint (single-backbone only)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("data/HumanML3D_official")
    text_dir = Path(cfg.paths.texts_dir)
    our_mean = np.load(out_dir / "Mean.npy").astype(np.float32)
    our_std = np.load(out_dir / "Std.npy").astype(np.float32)
    eval_mean, eval_std = load_eval_stats(cfg.paths.eval_stats_dir)
    motion_matcher, text_matcher = load_matchers(cfg.paths.eval_matcher, device=device)
    w_vec = WordVectorizer(r"data/t2m_glove/glove", "our_vab")
    build_text = make_build_text(w_vec)

    tokenizer = ResidualFsqTokenizer(cfg.tokenizer)
    tokenizer.load_state_dict(torch.load("checkpoints/tokenizer_fsq.pt", map_location="cpu"))
    tokenizer.to(device).eval()
    te = CLIPTextEncoder(cfg.text_encoder).to(device).eval()

    ids = [n.strip() for n in (out_dir / f"{args.split}.txt").read_text().splitlines() if n.strip()]
    ids = list(dict.fromkeys(i[1:] if i.startswith("M") else i for i in ids))

    backbones = ["transformer", "mamba"] if args.backbone == "both" else [args.backbone]
    print(
        f"device {device}  {args.split} ids {len(ids)}  cfg_scale {args.cfg_scale}  "
        f"temp {args.temperature}  (20-rep, our_vab)"
    )
    for backbone in backbones:
        ckpt = args.ckpt if len(backbones) == 1 else None
        if not Path(ckpt or f"checkpoints/generator_{backbone}.pt").is_file():
            print(f"{backbone:12s} SKIP (no checkpoint yet)")
            continue
        gt_feats, gen_feats, tok_lists = generate(
            backbone,
            cfg,
            tokenizer,
            te,
            device,
            ids,
            out_dir,
            text_dir,
            our_mean,
            our_std,
            args.max_clips,
            args.temperature,
            args.top_p,
            cfg_scale=args.cfg_scale,
            ckpt=ckpt,
        )
        evaluate(
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


if __name__ == "__main__":
    main()
