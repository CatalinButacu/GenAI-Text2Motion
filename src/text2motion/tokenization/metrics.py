from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from text2motion.motion.contracts import Split
from text2motion.motion.datasets import check_no_missing_files
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import JOINTS, recover_from_ric
from text2motion.motion.storage import FEATURES_DIR, base_clip_ids
from text2motion.tokenization.nn_model import TokenizerNetwork

type ReconstructionFidScorer = Callable[[list[np.ndarray], list[np.ndarray]], float]

EVALUATION_BATCH_CLIPS = 64


def select_reconstruction_evaluation_clip_ids(
    root: Path,
    split: Split | str,
    max_clips: int | None = None,
    shuffle_seed: int | None = None,
) -> list[str]:
    ids = base_clip_ids(root, split)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(ids)
    return ids[:max_clips]


@torch.no_grad()
def evaluate_tokenizer_reconstruction(
    module: TokenizerNetwork,
    root: Path,
    scaler: MotionScaler,
    max_frames: int,
    downsample: int | None = None,
    joint_count: int = JOINTS,
    device: str = "cpu",
    max_clips: int | None = None,
    *,
    split: Split | str,
    shuffle_seed: int | None = None,
    reconstruction_fid_scorer: ReconstructionFidScorer | None = None,
) -> dict[str, float]:
    if downsample is not None and downsample != module.downsample:
        raise ValueError(
            f"downsample {downsample} differs from the network downsample {module.downsample}"
        )
    was_training = module.training
    module.eval()
    try:
        return __reconstruction_metrics(
            module,
            root,
            scaler,
            max_frames,
            module.downsample,
            joint_count,
            device,
            max_clips,
            split=split,
            shuffle_seed=shuffle_seed,
            reconstruction_fid_scorer=reconstruction_fid_scorer,
        )
    finally:
        module.train(was_training)


def __reconstruction_metrics(
    module: TokenizerNetwork,
    root: Path,
    scaler: MotionScaler,
    max_frames: int,
    downsample: int,
    joint_count: int,
    device: str,
    max_clips: int | None,
    *,
    split: Split | str,
    shuffle_seed: int | None = None,
    reconstruction_fid_scorer: ReconstructionFidScorer | None = None,
) -> dict[str, float]:
    ids = select_reconstruction_evaluation_clip_ids(
        root, split, max_clips=max_clips, shuffle_seed=shuffle_seed
    )
    feature_paths = [root / FEATURES_DIR / f"{clip_id}.npy" for clip_id in ids]
    check_no_missing_files("feature", [str(path) for path in feature_paths if not path.is_file()])

    clips_by_length: dict[int, list[np.ndarray]] = {}
    non_finite: list[str] = []
    for path in feature_paths:
        features = np.load(path).astype(np.float32)
        if not np.isfinite(features).all():
            non_finite.append(path.stem)
            continue
        length = (min(features.shape[0], max_frames) // downsample) * downsample
        if length < 2 * downsample:
            continue
        clips_by_length.setdefault(length, []).append(features[:length])
    if non_finite:
        raise ValueError(f"{len(non_finite)} clips hold NaN or Inf values: {non_finite[:20]}")
    if not clips_by_length:
        raise ValueError(
            f"no clip of split {split} has {2 * downsample} usable frames "
            f"(clips listed: {len(ids)}, max_frames: {max_frames}); the metrics would be NaN"
        )

    batches = [
        clips[start : start + EVALUATION_BATCH_CLIPS]
        for clips in clips_by_length.values()
        for start in range(0, len(clips), EVALUATION_BATCH_CLIPS)
    ]
    reference_features: list[np.ndarray] = []
    reconstructed_features: list[np.ndarray] = []
    joint_errors: list[float] = []
    feature_distances: list[float] = []

    for batch in batches:
        reference = np.stack(batch)
        normalized = torch.from_numpy(scaler.normalize(reference)).to(device)
        reconstruction_normalized = module(normalized)[0].cpu().numpy()
        if reconstruction_normalized.shape != normalized.shape:
            raise ValueError(
                f"the reconstruction has shape {reconstruction_normalized.shape}, the input "
                f"{tuple(normalized.shape)}"
            )
        if not np.isfinite(reconstruction_normalized).all():
            raise ValueError("the reconstruction holds NaN or Inf values")
        reconstruction = np.asarray(scaler.denormalize(reconstruction_normalized), np.float32)

        reference_features.extend(reference)
        reconstructed_features.extend(reconstruction)
        feature_error = np.sqrt(((reference - reconstruction) ** 2).sum(-1)).mean(-1)
        feature_distances.extend(feature_error.tolist())

        reference_joints = recover_from_ric(torch.from_numpy(reference), joint_count)
        reconstructed_joints = recover_from_ric(torch.from_numpy(reconstruction), joint_count)
        joint_offsets = (reference_joints - reconstructed_joints).numpy()
        joint_errors.extend(np.sqrt((joint_offsets**2).sum(-1)).mean((-2, -1)).tolist())

    metrics = {
        "clips": len(reference_features),
        "mpjpe_mm": float(np.mean(joint_errors) * 1000.0),
        "feature_l2": float(np.mean(feature_distances)),
    }

    if reconstruction_fid_scorer is not None:
        metrics["recon_fid"] = reconstruction_fid_scorer(reference_features, reconstructed_features)
        if not np.isfinite(metrics["recon_fid"]):
            raise ValueError(f"the reconstruction FID is not finite: {metrics['recon_fid']}")

    return metrics
