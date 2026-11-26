from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from text2motion.evaluation.guo.encoders import GuoMotionEmbeddingEncoder, GuoTextEmbeddingEncoder
from text2motion.motion.representation import DIM


def load_guo_motion_normalization_stats(stats_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(Path(stats_dir) / "mean.npy")
    std = np.load(Path(stats_dir) / "std.npy")

    if mean.shape[-1] != DIM:
        raise ValueError(f"eval stats must be {DIM}-dim (standard HumanML3D), got {mean.shape}")

    return mean.astype(np.float32), std.astype(np.float32)


def load_frozen_guo_embedding_encoders(
    finest_tar: Path, device: str = "cpu"
) -> tuple[GuoMotionEmbeddingEncoder, GuoTextEmbeddingEncoder]:
    ckpt = torch.load(Path(finest_tar), map_location="cpu", weights_only=False)

    motion = GuoMotionEmbeddingEncoder()
    load_exact(motion.movement, ckpt["movement_encoder"], "movement_encoder")
    load_exact(motion.encoder, ckpt["motion_encoder"], "motion_encoder")

    text = GuoTextEmbeddingEncoder()
    load_exact(text, ckpt["text_encoder"], "text_encoder")

    return motion.to(device).eval(), text.to(device).eval()


def load_exact(module: nn.Module, state: dict, name: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(f"{name}: {len(missing)} params got no pretrained weight: {missing[:6]}")

    if unexpected:
        raise RuntimeError(f"{name}: {len(unexpected)} checkpoint keys unused: {unexpected[:6]}")
