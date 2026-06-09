"""Reconstruction evaluation for the motion tokenizer (Contribution A).

Encodes then decodes the held-out test motions and reports: MPJPE (mean per-joint position error in
mm, via ``recover_from_ric``), feature-L2, and the **downstream FID** of the reconstructions (the
Guo matcher's FID between GT and reconstructed motion embeddings). Downstream FID is the headline
number — it caps what any generator built on these tokens can reach. The tokenizer works in the
dataset's own Mean/Std space; FID uses the matcher's Comp_v6 normalization. See
``.claude/skills/motion-tokenizer`` and ``.claude/skills/t2m-eval``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.feature import recover_from_ric
from text2motion.eval.metrics import fid


def _embed_motions(matcher, feats_raw, eval_mean, eval_std, device, batch=32):
    embeddings = []
    for start in range(0, len(feats_raw), batch):
        group = feats_raw[start : start + batch]
        max_t = max(f.shape[0] for f in group)
        padded = np.zeros((len(group), max_t, 263), np.float32)
        lengths = [f.shape[0] for f in group]
        for row, feat in enumerate(group):
            padded[row, : feat.shape[0]] = (feat - eval_mean) / eval_std
        emb = matcher(torch.from_numpy(padded).to(device), torch.tensor(lengths, device=device))
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings)


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
) -> dict[str, float]:
    tokenizer.eval()
    ids = [n.strip() for n in (out_dir / "test.txt").read_text().splitlines() if n.strip()]
    ids = [i[1:] if i.startswith("M") else i for i in ids]
    ids = list(dict.fromkeys(ids))[:max_clips]

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

    gt_emb = _embed_motions(matcher, gt_feats, eval_mean, eval_std, device)
    recon_emb = _embed_motions(matcher, recon_feats, eval_mean, eval_std, device)

    return {
        "clips": len(gt_feats),
        "mpjpe_mm": float(np.mean(joint_errors) * 1000.0),
        "feature_l2": float(np.mean(feature_l2)),
        "recon_fid": fid(gt_emb, recon_emb),
    }
