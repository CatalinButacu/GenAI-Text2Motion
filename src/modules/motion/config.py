from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.shared.constants import (
    MOTION_DIM,
    SSM_D_MODEL,
    SSM_D_STATE,
    SSM_N_LAYERS,
)


@dataclass
class MotionConfig:
    """Runtime config: which checkpoints to load + how to post-process motion clips."""

    checkpointPath: str = "checkpoints/motion_ssm/best_model.pt"
    rvqCheckpointPath: str = "checkpoints/rvq_tokenizer/best_model.pt"
    blendFrames: int = 10
    minActionFrames: int = 20
    # Inference sampling: temperature=1.0+top_p=1.0 is greedy argmax (deterministic).
    # Raise temperature or lower top_p to explore diverse motions.
    temperature: float = 1.0
    topP: float = 1.0


@dataclass
class ModelConfig:
    """MotionSSM architecture hyperparameters."""

    dModel: int = SSM_D_MODEL
    dState: int = SSM_D_STATE
    nLayers: int = SSM_N_LAYERS
    motionDim: int = MOTION_DIM
    textEmbedDim: int = 256
    maxMotionLength: int = 200
    maxTextLength: int = 64
    vocabSize: int = 10000
    useSbert: bool = True
    sbertModel: str = "all-MiniLM-L6-v2"
    freezeSbert: bool = True
    bidirectional: bool = True
    gradientCheckpointing: bool = False
    useFilm: bool = True

    # --- RVQ head (Mogo/MoMask-style discrete token prediction) ---
    rvqLatentDim: int = 128
    rvqNCodebooks: int = 6
    rvqCodebookSize: int = 512
    rvqDownT: int = 4  # temporal stride: maxMotionLength / rvqDownT = latent seq len


@dataclass
class DataConfig:
    dataDir: str = "data/AMASS"
    amassDir: str = "data/AMASS"  # AMASS backing store used by HumanML3D loader
    arcticDataDir: str = "data/arctic/unpack"
    humanml3dDir: str = "data/humanml3d"
    interxDir: str = "data/inter-x"
    maxSamples: int | None = None
    numWorkers: int = field(
        default_factory=lambda: 0 if __import__("sys").platform == "win32" else 4
    )
    # Sources to use in unified training mode; subset of {"amass","arctic","humanml3d","interx"}
    unifiedSources: list = field(default_factory=lambda: ["amass", "arctic"])


@dataclass
class TrainingConfig(ModelConfig, DataConfig):
    batchSize: int = 32
    learningRate: float = 1e-4
    weightDecay: float = 0.01
    numEpochs: int = 200
    warmupSteps: int = 1000
    gradClip: float = 1.0
    lengthLossWeight: float = 0.1
    checkpointDir: str = "checkpoints/motion_ssm"
    rvqCheckpointPath: str = "checkpoints/rvq_tokenizer/best_model.pt"
    saveEvery: int = 10
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    resumeFrom: str | None = None
    seed: int | None = 42
    earlyStopPatience: int = 30
    keepLastCheckpoints: int = 5
    cfgDropoutProb: float = 0.1  # prob of dropping text conditioning at train time (CFG)
    useAmp: bool = True  # mixed precision on CUDA
