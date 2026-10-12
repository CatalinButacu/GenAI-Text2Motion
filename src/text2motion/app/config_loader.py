from __future__ import annotations

import os
from pathlib import Path

import yaml

from text2motion.app.config import (
    ApplicationConfig,
    ComputeDevice,
    Float32Precision,
    MotionServiceConfig,
    ProjectPathsConfig,
    SmplxAvatarConfig,
)
from text2motion.generation.contracts import (
    Backbone,
    GeneratorConfig,
    GeneratorTrainingConfig,
    LossWeighting,
    LossWeights,
    MixedPrecisionMode,
    TextEncoderConfig,
)
from text2motion.motion.contracts import MotionDataConfig
from text2motion.motion.model import Gender
from text2motion.motion.representation import DIM, FPS, JOINTS, MotionSource, RepresentationTrack
from text2motion.tokenization.contracts import (
    FsqComposition,
    RvqConfig,
    TokenizerConfig,
    TokenizerKind,
)

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


def __to_paths(raw: dict, keys: set[str]) -> dict:
    values = dict(raw)
    for key in keys:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return values


def __env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def __paths_config(raw: dict) -> ProjectPathsConfig:
    path_values = raw.get("paths", {})
    scalar_path_keys = set(ProjectPathsConfig.__dataclass_fields__) - {"generator_checkpoint_paths"}
    values = __to_paths(path_values, scalar_path_keys)
    if "generator_checkpoint_paths" in values:
        values["generator_checkpoint_paths"] = {
            Backbone(key).value: Path(value)
            for key, value in values["generator_checkpoint_paths"].items()
        }
    overrides = {
        "donor_root": __env_path("TEXT2MOTION_DONOR_ROOT"),
        "hml3d_out_dir": __env_path("TEXT2MOTION_DATASET_DIR"),
        "eval_matcher": __env_path("TEXT2MOTION_EVAL_MATCHER"),
        "eval_stats_dir": __env_path("TEXT2MOTION_EVAL_STATS_DIR"),
        "texts_dir": __env_path("TEXT2MOTION_TEXTS_DIR"),
        "smplx_models": __env_path("TEXT2MOTION_SMPLX_MODELS"),
        "checkpoints_dir": __env_path("TEXT2MOTION_CHECKPOINTS_DIR"),
        "outputs_dir": __env_path("TEXT2MOTION_OUTPUTS_DIR"),
        "logs_dir": __env_path("TEXT2MOTION_LOGS_DIR"),
        "cache_dir": __env_path("TEXT2MOTION_CACHE_DIR"),
        "tokenizer_checkpoint_path": __env_path("TEXT2MOTION_TOKENIZER_CHECKPOINT"),
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return ProjectPathsConfig(**values)


def __service_config(raw: dict) -> MotionServiceConfig:
    values = dict(raw.get("service", {}))
    env_values = {
        "bind_host": os.getenv("TEXT2MOTION_BIND_HOST"),
        "connect_host": os.getenv("TEXT2MOTION_CONNECT_HOST")
        or os.getenv("TEXT2MOTION_SERVICE_HOST"),
        "port": os.getenv("TEXT2MOTION_SERVICE_PORT"),
        "idle_seconds": os.getenv("TEXT2MOTION_IDLE_SECONDS"),
    }
    values.update({key: value for key, value in env_values.items() if value is not None})
    for key in ("port", "idle_seconds"):
        if key in values:
            values[key] = int(values[key])
    return MotionServiceConfig(**values)


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


def __avatar_config(raw: dict) -> SmplxAvatarConfig:
    values = dict(raw.get("avatar", {}))
    if "pose_segments" in values:
        values["pose_segments"] = tuple(
            (str(name), int(dimension)) for name, dimension in values["pose_segments"]
        )
    if "gender" in values:
        values["gender"] = Gender(values["gender"])
    return SmplxAvatarConfig(**__to_paths(values, {"models_path"}))


def __tokenizer_config(raw: dict) -> TokenizerConfig:
    values = dict(raw.get("tokenizer", {}))
    if "fsq_levels" in values:
        values["fsq_levels"] = tuple(int(level) for level in values["fsq_levels"])
    if "quantizer" in values:
        values["quantizer"] = FsqComposition(values["quantizer"])
    if "kind" in values:
        values["kind"] = TokenizerKind(values["kind"])
    return TokenizerConfig(**values)


def __data_config(raw: dict) -> MotionDataConfig:
    values = dict(raw.get("data", {}))
    if "track" in values:
        values["track"] = RepresentationTrack(values["track"])
    if "sources" in values:
        values["sources"] = tuple(MotionSource(source) for source in values["sources"])
    return MotionDataConfig(**values)


def __training_config(raw: dict) -> GeneratorTrainingConfig:
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
        values["loss_weights"] = LossWeights(**weights)
    if "amp" in values:
        values["amp"] = MixedPrecisionMode(values["amp"])
    if "loss_weighting" in values:
        values["loss_weighting"] = LossWeighting(values["loss_weighting"])
    return GeneratorTrainingConfig(**values)


def load_config(path: str | Path) -> ApplicationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    __check_representation_block(raw, path)
    generator_values = dict(raw.get("generator", {}))
    if "backbone" in generator_values:
        generator_values["backbone"] = Backbone(generator_values["backbone"])

    config = ApplicationConfig(
        seed=raw.get("seed", 2026),
        deterministic=raw.get("deterministic", True),
        device=ComputeDevice(raw.get("device", ComputeDevice.CUDA)),
        precision=Float32Precision(raw.get("precision", Float32Precision.HIGHEST)),
        avatar=__avatar_config(raw),
        paths=__paths_config(raw),
        service=__service_config(raw),
        data=__data_config(raw),
        tokenizer=__tokenizer_config(raw),
        rvq_baseline=RvqConfig(**raw.get("rvq_baseline", {})),
        text_encoder=TextEncoderConfig(**raw.get("text_encoder", {})),
        generator=GeneratorConfig(**generator_values),
        train=__training_config(raw),
    )
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
