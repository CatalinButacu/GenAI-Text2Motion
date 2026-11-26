from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from text2motion.dataset.contracts import MotionDataConfig
from text2motion.generation.contracts import (
    Backbone,
    GeneratorConfig,
    GeneratorTrainingConfig,
    TextEncoderConfig,
)
from text2motion.motion.contracts import Gender
from text2motion.tokenization.contracts import (
    ResidualVectorQuantizationConfig,
    TokenizerConfig,
)


############################################################
#       ComputeDevice
############################################################
class ComputeDevice(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


############################################################
#       Float32Precision
############################################################
class Float32Precision(StrEnum):
    HIGHEST = "highest"
    TF32 = "tf32"


############################################################
#       SmplxAvatarConfig
############################################################
@dataclass(frozen=True)
class SmplxAvatarConfig:
    model_type: str = "smplx"
    gender: Gender = Gender.NEUTRAL
    num_joints: int = 55
    num_vertices: int = 10475
    num_betas: int = 10
    use_pca: bool = False
    fps: int = 30
    up_axis: str = "z"
    models_path: Path | None = None
    pose_segments: tuple[tuple[str, int], ...] = (
        ("root_orient", 3),
        ("trans", 3),
        ("body_pose", 63),
        ("left_hand", 45),
        ("right_hand", 45),
        ("jaw_pose", 3),
        ("eyes_pose", 6),
    )

    @property
    def pose_dim(self) -> int:
        return sum(dim for _, dim in self.pose_segments)

    def slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        start = 0
        for name, dim in self.pose_segments:
            out[name] = slice(start, start + dim)
            start += dim
        return out


############################################################
#       ProjectPathsConfig
############################################################
@dataclass(frozen=True)
class ProjectPathsConfig:
    donor_root: Path | None = None
    humanml3d_dir: Path | None = None
    eval_matcher: Path | None = None
    eval_stats_dir: Path | None = None
    t2m_vqvae: Path | None = None
    glove_dir: Path | None = None
    amass_dir: Path | None = None
    smplx_models: Path | None = None
    stats_dir: Path | None = None
    smplh_dir: Path | None = None
    dmpl_dir: Path | None = None
    body_model_dir: Path | None = None
    hml3d_index_csv: Path | None = None
    hml3d_out_dir: Path | None = None
    texts_dir: Path | None = None
    our_vab_dir: Path = Path("data/models/glove/glove")
    cache_dir: Path = Path("data/.cache")
    checkpoints_dir: Path = Path("runs/checkpoints")
    outputs_dir: Path = Path("runs/outputs")
    logs_dir: Path = Path("runs/logs")
    tokenizer_checkpoint_path: Path | None = None
    generator_checkpoint_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def tokenizer_checkpoint(self) -> Path:
        if self.tokenizer_checkpoint_path is None:
            raise ValueError(
                "paths.tokenizer_checkpoint_path must be configured or passed explicitly"
            )
        return self.tokenizer_checkpoint_path

    def generator_checkpoint(self, backbone: Backbone | str) -> Path:
        key = Backbone(backbone).value
        if key not in self.generator_checkpoint_paths:
            raise ValueError(
                f"paths.generator_checkpoint_paths.{key} must be configured or passed explicitly"
            )
        return self.generator_checkpoint_paths[key]

    @property
    def inference_log(self) -> Path:
        return self.logs_dir / "perf" / "inference.jsonl"

    @property
    def service_log(self) -> Path:
        return self.logs_dir / "cli" / "motion_service.log"

    @property
    def benchmark_report(self) -> Path:
        return self.outputs_dir / "streaming_bench.json"


############################################################
#       MotionServiceConfig
############################################################
@dataclass(frozen=True)
class MotionServiceConfig:
    bind_host: str = "127.0.0.1"
    connect_host: str = "127.0.0.1"
    port: int = 8765
    idle_seconds: int = 600
    connect_timeout_seconds: float = 5.0
    startup_timeout_seconds: int = 240
    cuda_graph_decode: bool = False
    anchor_every_tokens: int = 0
    anchor_context_tokens: int = 16


############################################################
#       ApplicationConfig
############################################################
@dataclass(frozen=True)
class ApplicationConfig:
    seed: int = 2026
    deterministic: bool = True
    mlflow_tracking: bool = False
    device: ComputeDevice = ComputeDevice.CUDA
    precision: Float32Precision = Float32Precision.TF32
    failure_logs: bool = True
    detect_anomaly: bool = False
    avatar: SmplxAvatarConfig = field(default_factory=SmplxAvatarConfig)
    paths: ProjectPathsConfig = field(default_factory=ProjectPathsConfig)
    service: MotionServiceConfig = field(default_factory=MotionServiceConfig)
    data: MotionDataConfig = field(default_factory=MotionDataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    rvq_baseline: ResidualVectorQuantizationConfig = field(
        default_factory=ResidualVectorQuantizationConfig
    )
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    train: GeneratorTrainingConfig = field(default_factory=GeneratorTrainingConfig)
