from __future__ import annotations

import numpy as np
import torch

from text2motion.data.hml3d.dataset import parse_text_file
from text2motion.eval.context import EvalContext, GenerationPipeline, SamplingCfg
from text2motion.eval.embedding import embed_motions, embed_texts
from text2motion.eval.metrics import diversity, fid, mm_dist, r_precision


@torch.no_grad()
def evaluate_generation(
    pipeline: GenerationPipeline,
    ctx: EvalContext,
    sampling: SamplingCfg | None = None,
    max_clips: int | None = None,
    split: str = "val",  # in-train model selection MUST NOT touch test (only _twin_eval does, once)
) -> dict[str, float]:
    sampling = sampling or SamplingCfg()
    pipeline.eval()
    ids = ctx.clip_ids(split)[:max_clips]

    gt_feats, gen_feats, text_pairs = [], [], []
    for clip_id in ids:
        vec_path = ctx.out_dir / "new_joint_vecs" / f"{clip_id}.npy"
        text_path = ctx.text_dir / f"{clip_id}.txt"
        if not vec_path.is_file() or not text_path.is_file():
            continue
        feat = np.load(vec_path).astype(np.float32)
        token_len = min(feat.shape[0], 196) // ctx.downsample
        if token_len < 2:
            continue
        feat = feat[: token_len * ctx.downsample]
        caption_ann = parse_text_file(text_path)[0]
        gen = pipeline.generate(caption_ann.caption, token_len, sampling, ctx)
        if gen is None:
            continue

        gt_feats.append(feat)
        gen_feats.append(gen)
        text_pairs.append(ctx.build_text(caption_ann.tokens))

    gt_emb = embed_motions(ctx.motion_matcher, gt_feats, ctx.eval_mean, ctx.eval_std, ctx.device)
    gen_emb = embed_motions(ctx.motion_matcher, gen_feats, ctx.eval_mean, ctx.eval_std, ctx.device)
    text_emb_match = embed_texts(ctx.text_matcher, text_pairs, ctx.device)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(gen_emb))
    rp = r_precision(text_emb_match[perm], gen_emb[perm], pool_size=32, top_k=3)
    return {
        "clips": len(gen_feats),
        "fid": fid(gt_emb, gen_emb),
        "r_top1": float(rp[0]),
        "r_top3": float(rp[2]),
        "mm_dist": mm_dist(text_emb_match, gen_emb),
        "diversity": diversity(gen_emb),
    }
