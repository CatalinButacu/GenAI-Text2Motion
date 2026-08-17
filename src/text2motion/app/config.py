from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from text2motion.generation.losses import LossWeighting, LossWeights
from text2motion.generation.model import Backbone, GeneratorConfig
from text2motion.generation.text import TextEncoderConfig
from text2motion.generation.trainer import Amp, TrainingConfig
from text2motion.motion.dataset import DataConfig
from text2motion.motion.model import Gender
from text2motion.motion.representation import DIM, FPS, JOINTS, Source, Track
from text2motion.tokenization.model import (
    Quantizer,
    RvqConfig,
    TokenizerConfig,
    TokenizerKind,
)


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class Precision(StrEnum):
    HIGHEST = "highest"
    TF32 = "tf32"


LOSS_WEIGHT_KEYS = {
    "w_root": "root",
    "w_ric": "ric",
    "w_rot6d": "rot6d",
    "w_vel": "vel",
    "w_foot": "foot",
    "w_fk_self": "fk_self",
    "w_fk_gt": "fk_gt",
}


@dataclass(frozen=True)
class AvatarConfig:
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
class PathsConfig:
    donor_root: Path = Path(r"D:\Facultate\dissertation")
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
    checkpoints_dir: Path = Path("checkpoints")
    outputs_dir: Path = Path("outputs")


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    deterministic: bool = True
    device: Device = Device.CUDA
    precision: Precision = Precision.HIGHEST
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    rvq_baseline: RvqConfig = field(default_factory=RvqConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)


def _to_paths(raw: dict, keys: set[str]) -> dict:
    out = dict(raw)
    for key in keys:
        if out.get(key) is not None:
            out[key] = Path(out[key])
    return out


def _check_representation_block(raw: dict, source: str | Path) -> None:
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


def _avatar_config(raw: dict) -> AvatarConfig:
    avatar_raw = dict(raw.get("avatar", {}))
    if "pose_segments" in avatar_raw:
        avatar_raw["pose_segments"] = tuple(
            (str(name), int(dim)) for name, dim in avatar_raw["pose_segments"]
        )
    if "gender" in avatar_raw:
        avatar_raw["gender"] = Gender(avatar_raw["gender"])
    return AvatarConfig(**_to_paths(avatar_raw, {"models_path"}))


def _tokenizer_config(raw: dict) -> TokenizerConfig:
    tokenizer_raw = dict(raw.get("tokenizer", {}))
    if "fsq_levels" in tokenizer_raw:
        tokenizer_raw["fsq_levels"] = tuple(int(level) for level in tokenizer_raw["fsq_levels"])
    if "quantizer" in tokenizer_raw:
        tokenizer_raw["quantizer"] = Quantizer(tokenizer_raw["quantizer"])
    if "kind" in tokenizer_raw:
        tokenizer_raw["kind"] = TokenizerKind(tokenizer_raw["kind"])
    return TokenizerConfig(**tokenizer_raw)


def _data_config(raw: dict) -> DataConfig:
    data_raw = dict(raw.get("data", {}))
    if "track" in data_raw:
        data_raw["track"] = Track(data_raw["track"])
    if "sources" in data_raw:
        data_raw["sources"] = tuple(Source(source) for source in data_raw["sources"])
    return DataConfig(**data_raw)


def _training_config(raw: dict) -> TrainingConfig:
    train_raw = dict(raw.get("train", {}))
    weights = {
        field_name: train_raw.pop(key)
        for key, field_name in LOSS_WEIGHT_KEYS.items()
        if key in train_raw
    }
    if weights:
        train_raw["loss_weights"] = LossWeights(**weights)
    if "amp" in train_raw:
        train_raw["amp"] = Amp(train_raw["amp"])
    if "loss_weighting" in train_raw:
        train_raw["loss_weighting"] = LossWeighting(train_raw["loss_weighting"])
    return TrainingConfig(**train_raw)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    _check_representation_block(raw, path)

    generator_raw = dict(raw.get("generator", {}))
    if "backbone" in generator_raw:
        generator_raw["backbone"] = Backbone(generator_raw["backbone"])

    config = Config(
        seed=raw.get("seed", 2026),
        deterministic=raw.get("deterministic", True),
        device=Device(raw.get("device", Device.CUDA)),
        precision=Precision(raw.get("precision", Precision.HIGHEST)),
        avatar=_avatar_config(raw),
        paths=PathsConfig(**_to_paths(raw.get("paths", {}), set(PathsConfig.__dataclass_fields__))),
        data=_data_config(raw),
        tokenizer=_tokenizer_config(raw),
        rvq_baseline=RvqConfig(**raw.get("rvq_baseline", {})),
        text_encoder=TextEncoderConfig(**raw.get("text_encoder", {})),
        generator=GeneratorConfig(**generator_raw),
        train=_training_config(raw),
    )
    validate_config(config, path)
    return config


def validate_config(config: Config, source: str | Path = "<config>") -> None:
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
