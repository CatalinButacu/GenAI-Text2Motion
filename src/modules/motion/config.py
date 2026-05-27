from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import torch
import yaml

from src.shared.constants import (
    MOTION_DIM,
    SSM_D_MODEL,
    SSM_D_STATE,
    SSM_N_LAYERS,
)


def load_yaml_config(path: str | Path) -> dict:
    """Load a motion_ssm YAML and flatten its `architecture/rvq/training` sections.

    The YAML is grouped into sections for human readability; the loader returns
    a single flat dict that maps directly onto :class:`TrainingConfig` kwargs.
    Unknown keys are dropped silently so that adding new YAML knobs in the
    future does not break older code paths that don't recognise them.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} did not parse as a dict")
    flat: dict = {}

    for section, body in data.items():
        if isinstance(body, dict):
            flat.update(body)
        else:
            flat[section] = body

    return flat


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
    # Classifier-free guidance scale at inference. 1.0 = off (vanilla conditional).
    # 2-4 typical for text-conditioned motion; higher = stronger text adherence,
    # lower diversity. Requires the SSM to have been trained with cfg_dropout_prob > 0
    # AND use_sbert=True (CFG needs the uncond text-encoder path).
    cfg_scale: float = 1.0


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
    # Decoder architecture:
    #   "independent" -- legacy K-classifier head (RVQMotionDecoder). All K
    #                    codebooks predicted independently from SSM features.
    #                    Backward-compatible with existing checkpoints.
    #   "residual_k"  -- autoregressive across K (ResidualKHead). Each codebook
    #                    conditions on the embedded sum of prior-codebook
    #                    tokens. Restores RVQ residual structure at inference.
    #                    REQUIRES RETRAINING -- different param set.
    arch: str = "independent"
    model_dropout: float = 0.0  # inter-layer dropout; 0.0 = off (backward-compat default)

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
    warmup_steps: int = 500  # fresh training: ~7 epochs to reach peak LR
    grad_clip: float = 1.0
    length_loss_weight: float = 10.0  # compensates for normalised len_loss ~1e-3 (raw ~54)
    # Label smoothing for token CE (T2M-GPT style). Prevents overconfidence on VQ codes.
    label_smoothing: float = 0.1
    # ---- Geometric losses: DISABLED for the SSM generation model ----
    # The SSM operates in token space (like T2M-GPT, CVPR 2023). Geometry is already
    # encoded in the RVQ codebook embeddings (tokenizer trained with recon+commit).
    # Adding geometric losses creates conflicting gradients (token-space CE vs
    # motion-space L1 via soft-decode). T2M-GPT uses CE only; MDM uses geometric
    # losses only because it operates directly in motion space -- different paradigm.
    # Soft-decode path is skipped entirely when all weights are 0.0 (~15% step speedup).
    recon_loss_weight: float = 0.0
    velocity_loss_weight: float = 0.0
    root_height_loss_weight: float = 0.0
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
    # When False (default), the trainer auto-wraps in nn.DataParallel across all
    # visible CUDA devices. Set True to force single-GPU even with multiple
    # devices available (debugging, single-GPU baselining).
    single_gpu: bool = False
    # Pose-prefix curriculum (decision #11/#15). With this probability per
    # batch, the trainer splits each clip into (prefix, suffix), encodes the
    # prefix through the frozen tokenizer, projects via model.seed_from_latent,
    # and supervises prediction of the suffix only. Default 0.0 keeps the
    # cold-start training behaviour untouched. 0.5 is the recommended value
    # for the streaming-headline run (matches the action-transition rate the
    # demo will exercise at inference).
    pose_prefix_prob: float = 0.0
    # When True, the RVQ tokenizer trainer (NOT this script -- see
    # train_rvq_tokenizer.py) builds a causal decoder using CausalConv1d +
    # nearest-neighbor upsampling instead of ConvTranspose1d. Required for
    # streaming-mode inference; default False keeps existing checkpoints
    # bit-compatible.
    causal_decoder: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Build a TrainingConfig from configs/motion_ssm.yaml (or a smoke variant).

        Only keys that name an existing field are consumed; unknown keys are
        ignored so the YAML can carry forward-looking knobs without crashing
        older code. CLI flags should override the resulting config explicitly
        in the caller (the YAML is the *base*, not the final word).
        """
        flat = load_yaml_config(path)
        valid = {f.name for f in fields(cls)}
        consumed = {k: v for k, v in flat.items() if k in valid}

        return cls(**consumed)
