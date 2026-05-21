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

    checkpoint_path: str = "checkpoints/motion_ssm/best_model.pt"
    rvq_checkpoint_path: str = "checkpoints/rvq_tokenizer/best_model.pt"
    blend_frames: int = 10
    min_action_frames: int = 20
    # Inference sampling: temperature=1.0+top_p=1.0 is greedy argmax (deterministic).
    # Raise temperature or lower top_p to explore diverse motions.
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class ModelConfig:
    """MotionSSM architecture hyperparameters."""

    d_model: int = SSM_D_MODEL
    d_state: int = SSM_D_STATE
    n_layers: int = SSM_N_LAYERS
    motion_dim: int = MOTION_DIM
    text_embed_dim: int = 256
    max_motion_length: int = 200
    max_text_length: int = 64
    vocab_size: int = 10000
    use_sbert: bool = True
    sbert_model: str = "all-MiniLM-L6-v2"
    freeze_sbert: bool = True
    bidirectional: bool = True
    gradient_checkpointing: bool = False
    use_film: bool = True

    # --- RVQ head (Mogo/MoMask-style discrete token prediction) ---
    rvq_latent_dim: int = 128
    rvq_n_codebooks: int = 6
    rvq_codebook_size: int = 512
    rvq_down_t: int = 4  # temporal stride: max_motion_length / rvq_down_t = latent seq len


@dataclass
class DataConfig:
    data_dir: str = "data/AMASS"
    amass_dir: str = "data/AMASS"  # AMASS backing store used by HumanML3D loader
    arctic_data_dir: str = "data/arctic/unpack"
    humanml3d_dir: str = "data/humanml3d"
    interx_dir: str = "data/inter-x"
    max_samples: int | None = None
    num_workers: int = field(
        default_factory=lambda: 0 if __import__("sys").platform == "win32" else 4
    )
    # Sources to use in unified training mode; subset of {"amass","arctic","humanml3d","interx"}
    unified_sources: list = field(default_factory=lambda: ["amass", "arctic"])


@dataclass
class TrainingConfig(ModelConfig, DataConfig):
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 200
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    length_loss_weight: float = 0.1
    checkpoint_dir: str = "checkpoints/motion_ssm"
    rvq_checkpoint_path: str = "checkpoints/rvq_tokenizer/best_model.pt"
    save_every: int = 10
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    resume_from: str | None = None
    # Warm start: when resuming, load only model weights (not optimizer / scheduler /
    # epoch counter / step counter). Use this to change LR or other hyperparams while
    # keeping the pretrained weights as initialisation. best_loss is still inherited so
    # the early-stop "improvement" bar is the prior best.
    warm_start: bool = False
    seed: int | None = 42
    early_stop_patience: int = 30
    keep_last_checkpoints: int = 5
    cfg_dropout_prob: float = 0.1  # prob of dropping text conditioning at train time (CFG)
    use_amp: bool = True  # mixed precision on CUDA
    # Wrap the model in torch.compile(mode="reduce-overhead", dynamic=False).
    # Expected 3-5x training throughput on GPU; first batch eats 30-90s of compile.
    # CPU benchmarks show compile is usually slower due to dispatch overhead,
    # so this stays opt-in and gated on CUDA in the trainer.
    compile_model: bool = False
