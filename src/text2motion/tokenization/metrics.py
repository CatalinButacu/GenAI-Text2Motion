from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import JOINTS, recover_from_ric
from text2motion.motion.storage import FEATURES_DIR
from text2motion.tokenization.nn_model import TokenizerNetwork

type ReconstructionFidScorer = Callable[[list[np.ndarray], list[np.ndarray]], float]


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
    module: TokenizerNetwork,
    root: Path,
    scaler: MotionScaler,
    max_frames: int,
    downsample: int = 4,
    joint_count: int = JOINTS,
    device: str = "cpu",
    max_clips: int | None = None,
    split: Split | str = Split.TEST,
    shuffle_seed: int | None = None,
    reconstruction_fid_scorer: ReconstructionFidScorer | None = None,
) -> dict[str, float]:
    module.eval()
    ids = select_reconstruction_evaluation_clip_ids(
        root, split, max_clips=max_clips, shuffle_seed=shuffle_seed
    )

    reference_features: list[np.ndarray] = []
    reconstructed_features: list[np.ndarray] = []
    joint_errors: list[float] = []
    feature_distances: list[float] = []

    for clip_id in ids:
        path = root / FEATURES_DIR / f"{clip_id}.npy"
        if not path.is_file():
            continue
        features = np.load(path).astype(np.float32)
        length = (min(features.shape[0], max_frames) // downsample) * downsample
        if length < 2 * downsample:
            continue
        features = features[:length]

        normalized = torch.from_numpy(scaler.normalize(features))[None].to(device)
        reconstruction_normalized = module(normalized)[0][0].cpu().numpy()
        reconstruction = np.asarray(scaler.denormalize(reconstruction_normalized))

        reference_features.append(features)
        reconstructed_features.append(reconstruction)
        feature_error = np.sqrt(((features - reconstruction) ** 2).sum(-1)).mean()
        feature_distances.append(float(feature_error))

        reference_joints = recover_from_ric(torch.from_numpy(features).float(), joint_count)
        reconstructed_joints = recover_from_ric(
            torch.from_numpy(reconstruction).float(), joint_count
        )
        joint_offsets = reference_joints.numpy() - reconstructed_joints.numpy()
        joint_errors.append(float(np.sqrt((joint_offsets**2).sum(-1)).mean()))

    metrics = {
        "clips": len(reference_features),
        "mpjpe_mm": float(np.mean(joint_errors) * 1000.0),
        "feature_l2": float(np.mean(feature_distances)),
    }

    if reconstruction_fid_scorer is not None:
        metrics["recon_fid"] = reconstruction_fid_scorer(reference_features, reconstructed_features)

    return metrics
