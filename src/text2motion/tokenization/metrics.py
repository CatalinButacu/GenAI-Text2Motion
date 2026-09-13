from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import JOINTS, recover_from_ric
from text2motion.motion.storage import FEATURES_DIR
from text2motion.tokenization.model import MotionTokenizerNetwork

_ReconFidScorer = Callable[[list[np.ndarray], list[np.ndarray]], float]


def select_reconstruction_evaluation_clip_ids(
    root: Path,
    split: Split | str,
    max_clips: int | None = None,
    shuffle_seed: int | None = None,
) -> list[str]:
    listed = [n.strip() for n in (root / f"{split}.txt").read_text().splitlines() if n.strip()]
    ids = list(dict.fromkeys(i[1:] if i.startswith("M") else i for i in listed))
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(ids)
    return ids[:max_clips]


@torch.no_grad()
def evaluate_tokenizer_reconstruction(
    module: MotionTokenizerNetwork,
    root: Path,
    scaler: MotionScaler,
    max_frames: int,
    downsample: int = 4,
    joint_count: int = JOINTS,
    device: str = "cpu",
    max_clips: int | None = None,
    split: Split | str = Split.TEST,
    shuffle_seed: int | None = None,
    recon_fid: _ReconFidScorer | None = None,
) -> dict[str, float]:
    module.eval()
    ids = select_reconstruction_evaluation_clip_ids(root, split, max_clips=max_clips, shuffle_seed=shuffle_seed)

    reference_feats: list[np.ndarray] = []
    recon_feats: list[np.ndarray] = []
    joint_errors: list[float] = []
    feature_l2: list[float] = []

    for clip_id in ids:
        path = root / FEATURES_DIR / f"{clip_id}.npy"
        if not path.is_file():
            continue
        feat = np.load(path).astype(np.float32)
        length = (min(feat.shape[0], max_frames) // downsample) * downsample
        if length < 2 * downsample:
            continue
        feat = feat[:length]

        normalized = torch.from_numpy(scaler.normalize(feat))[None].to(device)
        recon = scaler.denormalize(module(normalized)[0][0].cpu().numpy())

        reference_feats.append(feat)
        recon_feats.append(recon)
        feature_l2.append(float(np.sqrt(((feat - recon) ** 2).sum(-1)).mean()))

        reference_joints = recover_from_ric(torch.from_numpy(feat).float(), joint_count).numpy()
        recon_joints = recover_from_ric(torch.from_numpy(recon).float(), joint_count).numpy()
        joint_errors.append(float(np.sqrt(((reference_joints - recon_joints) ** 2).sum(-1)).mean()))

    metrics = {
        "clips": len(reference_feats),
        "mpjpe_mm": float(np.mean(joint_errors) * 1000.0),
        "feature_l2": float(np.mean(feature_l2)),
    }

    if recon_fid is not None:
        metrics["recon_fid"] = recon_fid(reference_feats, recon_feats)

    return metrics
