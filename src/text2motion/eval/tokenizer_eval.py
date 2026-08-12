from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.feature import recover_from_ric
from text2motion.eval.embedding import embed_motions
from text2motion.eval.metrics import fid


@torch.no_grad()
def evaluate_tokenizer(
    tokenizer,
    out_dir: Path,
    our_mean: np.ndarray,
    our_std: np.ndarray,
    matcher,
    eval_mean: np.ndarray,
    eval_std: np.ndarray,
    joints_num: int = 22,
    device: str = "cpu",
    max_clips: int | None = None,
    split: str = "test",
    shuffle_seed: int | None = None,
) -> dict[str, float]:
    tokenizer.eval()
    ids = [n.strip() for n in (out_dir / f"{split}.txt").read_text().splitlines() if n.strip()]
    ids = [i[1:] if i.startswith("M") else i for i in ids]
    ids = list(dict.fromkeys(ids))
    if shuffle_seed is not None:  # representative equal-size sample for cross-split gap comparison
        np.random.default_rng(shuffle_seed).shuffle(ids)
    ids = ids[:max_clips]

    gt_feats, recon_feats = [], []
    joint_errors, feature_l2 = [], []
    for clip_id in ids:
        path = out_dir / "new_joint_vecs" / f"{clip_id}.npy"
        if not path.is_file():
            continue
        feat = np.load(path).astype(np.float32)
        length = (min(feat.shape[0], 196) // 4) * 4
        if length < 8:
            continue
        feat = feat[:length]

        normalized = torch.from_numpy((feat - our_mean) / our_std)[None].to(device)
        recon = tokenizer(normalized)[0][0].cpu().numpy() * our_std + our_mean

        gt_feats.append(feat)
        recon_feats.append(recon)
        feature_l2.append(float(np.sqrt(((feat - recon) ** 2).sum(-1)).mean()))

        gt_joints = recover_from_ric(torch.from_numpy(feat).float(), joints_num).numpy()
        rc_joints = recover_from_ric(torch.from_numpy(recon).float(), joints_num).numpy()
        joint_errors.append(float(np.sqrt(((gt_joints - rc_joints) ** 2).sum(-1)).mean()))

    gt_emb = embed_motions(matcher, gt_feats, eval_mean, eval_std, device)
    recon_emb = embed_motions(matcher, recon_feats, eval_mean, eval_std, device)

    return {
        "clips": len(gt_feats),
        "mpjpe_mm": float(np.mean(joint_errors) * 1000.0),
        "feature_l2": float(np.mean(feature_l2)),
        "recon_fid": fid(gt_emb, recon_emb),
    }
