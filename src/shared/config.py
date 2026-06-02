from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch
import yaml

from src.shared.constants import CONSTS, MAMBA, RVQ, SMPLX, SSM

# ── YAML loader ────────────────────────────────────────────────────────────────

def load_yaml_config(path: str | Path) -> dict:
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


# ── Text understanding ─────────────────────────────────────────────────────────

@dataclass
class SpacyConfig:
    model: str = "en_core_web_sm"


@dataclass
class ParserConfig:
    spacy: SpacyConfig = field(default_factory=SpacyConfig)
    use_spacy: bool = True


# ── Spatial layout ─────────────────────────────────────────────────────────────

@dataclass
class PlannerConfig:
    random_layout: bool = False
    base_duration: float = 5.0
    duration_jitter: float = 0.0
    actor_dist: float = 1.5
    obj_space: float = 0.4
    fall_height: float = 1.5
    ground_height: float = 0.5
    default_size: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    default_mass: float = 1.0
    close_action_dist: dict[str, float] = field(
        default_factory=lambda: {"kick": 0.8, "pick_up": 0.5}
    )
    random_range: float = 2.0
    random_seed: int | None = None


# ── Rendering ──────────────────────────────────────────────────────────────────

@dataclass
class RenderConfig:
    fps: int = 30
    gender: str = "neutral"
    width: int = 1280
    height: int = 720
    num_betas: int = 10


# ── Motion inference ───────────────────────────────────────────────────────────

@dataclass
class MotionConfig:
    checkpoint_path: str = "checkpoints/motion_ssm/best_model.pt"
    rvq_checkpoint_path: str = "checkpoints/rvq_tokenizer/best_model.pt"
    blend_frames: int = 10
    init_pose_blend_frames: int = 10
    min_action_frames: int = 20
    temperature: float = 1.0
    top_p: float = 1.0
    cfg_scale: float = 2.0
    use_film: bool = True
    bidirectional: bool = True
    rerank: bool = False
    num_candidates: int = 4
    use_retrieval: bool = False
    retrieval_index_path: str = "data/retrieval_index.npz"
    retrieval_top_k: int = 3


# ── Model architecture ─────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    d_model: int = SSM.d_model
    d_state: int = SSM.d_state
    n_layers: int = SSM.n_layers
    motion_dim: int = SMPLX.pose_dim
    text_embed_dim: int = 256
    max_motion_length: int = 200
    max_text_length: int = 64
    vocab_size: int = 10000
    use_sbert: bool = True
    sbert_model: str = "all-MiniLM-L6-v2"
    freeze_sbert: bool = True
    bidirectional: bool = False
    gradient_checkpointing: bool = False
    use_film: bool = True
    arch: str = "residual_k"
    model_dropout: float = 0.0
    rvq_latent_dim: int = 128
    rvq_n_codebooks: int = 6
    rvq_codebook_size: int = 512
    rvq_down_t: int = 4


# ── Data loading ───────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    data_dir: str = "data/AMASS"
    amass_dir: str = "data/AMASS"
    arctic_data_dir: str = "data/arctic/unpack"
    humanml3d_dir: str = "data/humanml3d"
    interx_dir: str = "data/inter-x"
    max_samples: int | None = None
    num_workers: int = field(default_factory=lambda: 0 if sys.platform == "win32" else 4)
    unified_sources: list = field(default_factory=lambda: ["amass", "arctic"])


# ── SSM training ───────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig(ModelConfig, DataConfig):
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 200
    warmup_steps: int = 500
    grad_clip: float = 1.0
    length_loss_weight: float = 10.0
    label_smoothing: float = 0.1
    recon_loss_weight: float = 0.0
    velocity_loss_weight: float = 0.0
    root_height_loss_weight: float = 0.0
    checkpoint_dir: str = "checkpoints/motion_ssm"
    rvq_checkpoint_path: str = "checkpoints/rvq_tokenizer/best_model.pt"
    save_every: int = 10
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    resume_from: str | None = None
    warm_start: bool = False
    seed: int | None = 42
    early_stop_patience: int = 30
    keep_last_checkpoints: int = 5
    cfg_dropout_prob: float = 0.1
    use_amp: bool = True
    compile_model: bool = False
    single_gpu: bool = False
    # pose_prefix_prob: not consumed by the current trainer (only by model API).
    # Keep at 0.0; train_motion_ssm raises if a user sets a positive value.
    pose_prefix_prob: float = 0.0
    causal_decoder: bool = True
    # Adapter / LoRA fine-tune mode. When True, the trainer freezes the SSM
    # trunk + text encoder + positional embeddings and only trains FiLM, the
    # RVQ decoder head, and condition_proj. Requires ``adapter_base_ckpt`` so
    # the frozen weights start from a trained baseline, not random init.
    adapter_mode: bool = False
    adapter_base_ckpt: str | None = None
    adapter_lora_targets: list = field(default_factory=list)  # e.g. ['condition_proj']
    adapter_lora_rank: int = 8

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        flat = load_yaml_config(path)
        valid = {f.name for f in fields(cls)}
        consumed = {k: v for k, v in flat.items() if k in valid}

        return cls(**consumed)


# ── RVQ tokenizer training ─────────────────────────────────────────────────────

@dataclass
class RvqTrainingConfig:
    data_dir: str = "data/AMASS"
    data_source: str = "amass"
    humanml3d_dir: str = "data/humanml3d"
    arctic_dir: str = "data/arctic/unpack"
    interx_dir: str = "data/inter-x"
    max_motion_length: int = 200
    max_samples_per_source: int | None = None
    stats_path: str | None = None
    trans_stats_path: str | None = None
    latent_dim: int = 128
    n_codebooks: int = 6
    codebook_size: int = 512
    down_t: int = 4
    epochs: int = 200
    batch_size: int = 32
    lr: float = 2e-4
    recon_weight: float = 1.0
    vel_weight: float = 0.5
    commit_weight: float = 0.25
    reset_dead_every: int = 5
    reset_dead_threshold: float = 1.0
    checkpoint_dir: str = "checkpoints/rvq_tokenizer"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    num_workers: int = field(default_factory=lambda: 0 if sys.platform == "win32" else 4)
    resume: str | None = None
    warm_start: bool = False
    wandb_mode: str = "offline"


# ── Pipeline ───────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    duration: float = 5.0
    fps: int = 30
    device: str = "cuda"
    prompt_max_chars: int = 2000
    seed: int = 42

    understanding: ParserConfig = field(default_factory=ParserConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def __post_init__(self) -> None:
        self.planner.base_duration = self.duration
        self.render.fps = self.fps

    def video_path(self, outputName: str) -> str:
        return os.path.join(self.output_dir, "videos", f"{outputName}.mp4")
