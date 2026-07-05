from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.dataset import parse_text_file
from text2motion.eval.metrics import diversity, fid, mm_dist, r_precision
from text2motion.eval.tokenizer_eval import _embed_motions


def _embed_texts(text_matcher, pairs, device, batch=32):
    embeddings = []
    for start in range(0, len(pairs), batch):
        group = pairs[start : start + batch]
        max_len = max(we.shape[0] for we, _ in group)
        we_pad = np.zeros((len(group), max_len, 300), np.float32)
        pe_pad = np.zeros((len(group), max_len, 15), np.float32)
        lengths = [we.shape[0] for we, _ in group]
        for row, (we, pe) in enumerate(group):
            we_pad[row, : we.shape[0]] = we
            pe_pad[row, : pe.shape[0]] = pe
        emb = text_matcher(
            torch.from_numpy(we_pad).to(device),
            torch.from_numpy(pe_pad).to(device),
            lengths=torch.tensor(lengths, device=device),
        )
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings)


@torch.no_grad()
def evaluate_generation(
    generator,
    text_encoder,
    tokenizer,
    out_dir: Path,
    text_dir: Path,
    our_mean: np.ndarray,
    our_std: np.ndarray,
    motion_matcher,
    text_matcher,
    build_text: Callable,
    eval_mean: np.ndarray,
    eval_std: np.ndarray,
    downsample: int = 4,
    device: str = "cpu",
    max_clips: int | None = None,
    temperature: float = 1.0,
    top_p: float = 0.9,
    cfg_scale: float = 1.0,
    split: str = "val",  # in-train model selection MUST NOT touch test (only _twin_eval does, once)
) -> dict[str, float]:
    generator.eval()
    text_encoder.eval()
    tokenizer.eval()

    ids = [n.strip() for n in (out_dir / f"{split}.txt").read_text().splitlines() if n.strip()]
    ids = list(dict.fromkeys(i[1:] if i.startswith("M") else i for i in ids))[:max_clips]

    gt_feats, gen_feats, text_pairs = [], [], []
    for clip_id in ids:
        vec_path = out_dir / "new_joint_vecs" / f"{clip_id}.npy"
        text_path = text_dir / f"{clip_id}.txt"
        if not vec_path.is_file() or not text_path.is_file():
            continue
        feat = np.load(vec_path).astype(np.float32)
        token_len = min(feat.shape[0], 196) // downsample
        if token_len < 2:
            continue
        feat = feat[: token_len * downsample]
        caption_ann = parse_text_file(text_path)[0]

        text_emb = text_encoder([caption_ann.caption])
        tokens = torch.stack(
            list(
                generator.stream(
                    text_emb, token_len, temperature=temperature, top_p=top_p, cfg_scale=cfg_scale
                )
            ),
            dim=1,
        )
        gen = tokenizer.decode(tokens)[0].cpu().numpy() * our_std + our_mean

        gt_feats.append(feat)
        gen_feats.append(gen)
        text_pairs.append(build_text(caption_ann.tokens))

    gt_emb = _embed_motions(motion_matcher, gt_feats, eval_mean, eval_std, device)
    gen_emb = _embed_motions(motion_matcher, gen_feats, eval_mean, eval_std, device)
    text_emb_match = _embed_texts(text_matcher, text_pairs, device)

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
