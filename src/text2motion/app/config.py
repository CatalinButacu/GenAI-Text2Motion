from __future__ import annotations

import difflib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import GenericAlias
from typing import Any, TypeVar, cast

import yaml
from pydantic import ConfigDict, TypeAdapter

from text2motion.generation.contracts import (
    Backbone,
    GeneratorConfig,
    GeneratorTrainingConfig,
    TextEncoderConfig,
)
from text2motion.motion.contracts import MotionDataConfig
from text2motion.motion.model import Gender
from text2motion.motion.representation import DIM, FPS, JOINTS
from text2motion.tokenization.contracts import RvqConfig, TokenizerConfig, TokenizerKind


class ComputeDevice(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class Float32Precision(StrEnum):
    HIGHEST = "highest"
    TF32 = "tf32"


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
    our_vab_dir: Path = Path("data/t2m_glove/glove")
    cache_dir: Path = Path("data/.cache")
    checkpoints_dir: Path = Path("artifacts/checkpoints")
    outputs_dir: Path = Path("artifacts/outputs")
    logs_dir: Path = Path("artifacts/logs")
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


@dataclass(frozen=True)
class ApplicationConfig:
    seed: int = 2026
    deterministic: bool = True
    mlflow_tracking: bool = False
    device: ComputeDevice = ComputeDevice.CUDA
    precision: Float32Precision = Float32Precision.HIGHEST
    avatar: SmplxAvatarConfig = field(default_factory=SmplxAvatarConfig)
    paths: ProjectPathsConfig = field(default_factory=ProjectPathsConfig)
    service: MotionServiceConfig = field(default_factory=MotionServiceConfig)
    data: MotionDataConfig = field(default_factory=MotionDataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    rvq_baseline: RvqConfig = field(default_factory=RvqConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    train: GeneratorTrainingConfig = field(default_factory=GeneratorTrainingConfig)


_T = TypeVar("_T")
_STRICT_KEYS = ConfigDict(extra="forbid")


def from_mapping(cls: type[_T], data: Mapping[str, Any]) -> _T:
    single = cast("type[tuple[_T]]", GenericAlias(tuple, (cls,)))
    strict_single = TypeAdapter(single, config=_STRICT_KEYS)
    return strict_single.validate_python((dict(data),))[0]


def load_service_config(path: str | Path) -> MotionServiceConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_mapping(MotionServiceConfig, __with_environment("service", {"service": raw}))


def service_policy(service: MotionServiceConfig) -> dict[str, object]:
    return {
        "anchor_every_tokens": service.anchor_every_tokens,
        "anchor_context_tokens": service.anchor_context_tokens,
        "cuda_graph_decode": service.cuda_graph_decode,
    }


def load_dataclass(cls: type[_T], path: str | Path) -> _T:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_mapping(cls, raw)


_LEGACY_ROOT_KEYS = {"hml3d"}
_LOSS_WEIGHT_KEYS = {
    "w_root": "root",
    "w_ric": "ric",
    "w_rot6d": "rot6d",
    "w_vel": "vel",
    "w_foot": "foot",
    "w_fk_self": "fk_self",
    "w_fk_gt": "fk_gt",
    "w_foot_skate": "foot_skate",
    "w_accel": "accel",
}
_ENVIRONMENT_OVERRIDES = {
    "paths": {
        "donor_root": ("TEXT2MOTION_DONOR_ROOT",),
        "hml3d_out_dir": ("TEXT2MOTION_DATASET_DIR",),
        "eval_matcher": ("TEXT2MOTION_EVAL_MATCHER",),
        "eval_stats_dir": ("TEXT2MOTION_EVAL_STATS_DIR",),
        "texts_dir": ("TEXT2MOTION_TEXTS_DIR",),
        "smplx_models": ("TEXT2MOTION_SMPLX_MODELS",),
        "checkpoints_dir": ("TEXT2MOTION_CHECKPOINTS_DIR",),
        "outputs_dir": ("TEXT2MOTION_OUTPUTS_DIR",),
        "logs_dir": ("TEXT2MOTION_LOGS_DIR",),
        "cache_dir": ("TEXT2MOTION_CACHE_DIR",),
        "tokenizer_checkpoint_path": ("TEXT2MOTION_TOKENIZER_CHECKPOINT",),
    },
    "service": {
        "bind_host": ("TEXT2MOTION_BIND_HOST",),
        "connect_host": ("TEXT2MOTION_CONNECT_HOST", "TEXT2MOTION_SERVICE_HOST"),
        "port": ("TEXT2MOTION_SERVICE_PORT",),
        "idle_seconds": ("TEXT2MOTION_IDLE_SECONDS",),
    },
}


def __with_environment(section: str, raw: dict) -> dict:
    values = dict(raw.get(section, {}))
    for key, names in _ENVIRONMENT_OVERRIDES[section].items():
        value = next((os.getenv(name) for name in names if os.getenv(name)), None)
        if value is not None:
            values[key] = value
    return values


def __check_representation_block(raw: dict, source: str | Path) -> None:
    declared = raw.get("hml3d")
    if not declared:
        return
    expected = {"dim": DIM, "num_joints": JOINTS, "fps": FPS}
    for key, value in expected.items():
        if key in declared and int(declared[key]) != value:
            raise ValueError(
                f"{source}: hml3d.{key} is {declared[key]}, but the HumanML3D-263 representation "
                f"contract fixes it at {value}. The layout is a static domain contract "
                f"(motion/representation.py), not a per-run knob -- every trained checkpoint, the "
                f"Mean/Std scaler, and the Guo evaluator all assume it."
            )


def __training_values(raw: dict) -> dict:
    values = dict(raw.get("train", {}))
    if "loss_weights" in values:
        raise ValueError(
            "train.loss_weights is not a config key; set the weights as flat train keys: "
            + ", ".join(sorted(_LOSS_WEIGHT_KEYS))
        )
    weights = {
        field_name: values.pop(key)
        for key, field_name in _LOSS_WEIGHT_KEYS.items()
        if key in values
    }
    if weights:
        values["loss_weights"] = weights
    return values


def __check_root_keys(raw: dict, source: str | Path) -> None:
    known = set(ApplicationConfig.__dataclass_fields__) | _LEGACY_ROOT_KEYS
    unknown = sorted(set(raw) - known)
    if not unknown:
        return
    hints = {key: difflib.get_close_matches(key, sorted(known), n=1) for key in unknown}
    described = ", ".join(
        f"{key!r} (did you mean {hint[0]!r}?)" if hint else repr(key) for key, hint in hints.items()
    )
    raise ValueError(f"{source}: unknown root keys {described}; known: {sorted(known)}")


def load_config(path: str | Path) -> ApplicationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    __check_root_keys(raw, path)
    __check_representation_block(raw, path)
    values = {key: raw[key] for key in ApplicationConfig.__dataclass_fields__ if key in raw}
    values["paths"] = __with_environment("paths", raw)
    values["service"] = __with_environment("service", raw)
    values["train"] = __training_values(raw)
    config = from_mapping(ApplicationConfig, values)
    validate_config(config, path)
    return config


def validate_config(config: ApplicationConfig, source: str | Path = "<config>") -> None:
    if config.generator.text_prefix_len != config.text_encoder.prefix_len:
        raise ValueError(
            f"{source}: generator.text_prefix_len ({config.generator.text_prefix_len}) must equal "
            f"text_encoder.prefix_len ({config.text_encoder.prefix_len}) -- the generator reserves "
            f"exactly the prefix positions the encoder emits. A silent mismatch changes the text "
            f"conditioning, which confounds any capacity comparison run against this config."
        )

    prefix_and_motion = config.generator.text_prefix_len + config.data.max_motion_len // (
        config.tokenizer.downsample or 1
    )
    if config.generator.max_seq_len < prefix_and_motion:
        raise ValueError(
            f"{source}: generator.max_seq_len ({config.generator.max_seq_len}) is smaller than "
            f"text_prefix_len + max_motion_len/downsample ({prefix_and_motion}); the longest "
            f"training clip would be truncated by the position budget."
        )

    if (
        config.tokenizer.kind == TokenizerKind.RVQ
        and config.tokenizer.num_quantizers != config.rvq_baseline.num_quantizers
    ):
        raise ValueError(
            f"{source}: tokenizer.num_quantizers ({config.tokenizer.num_quantizers}) must equal "
            f"rvq_baseline.num_quantizers ({config.rvq_baseline.num_quantizers}) -- the generator's "
            f"num_codebooks is resolved from tokenizer.num_quantizers, while the RVQ tokenizer model "
            f"itself is built from rvq_baseline.num_quantizers. A silent mismatch shapes the "
            f"generator's token embedding table wrong for the tokenizer it will actually decode."
        )
