from __future__ import annotations

import difflib
from pathlib import Path
from types import MappingProxyType

from text2motion.app.config.schema import ApplicationConfig
from text2motion.motion.representation import DIM, FPS, JOINTS
from text2motion.tokenization.contracts import (
    TokenizerKind,
)

_LEGACY_ROOT_KEYS = frozenset({"hml3d"})


_LOSS_WEIGHT_KEYS = MappingProxyType(
    {
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
)


def check_representation_block(raw: dict, source: str | Path) -> None:
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


def training_values(raw: dict) -> dict:
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


def check_root_keys(raw: dict, source: str | Path) -> None:
    known = set(ApplicationConfig.__dataclass_fields__) | _LEGACY_ROOT_KEYS
    unknown = sorted(set(raw) - known)
    if not unknown:
        return
    hints = {key: difflib.get_close_matches(key, sorted(known), n=1) for key in unknown}
    described = ", ".join(
        f"{key!r} (did you mean {hint[0]!r}?)" if hint else repr(key) for key, hint in hints.items()
    )
    raise ValueError(f"{source}: unknown root keys {described}; known: {sorted(known)}")


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
        config.tokenizer.kind == TokenizerKind.RESIDUAL_VECTOR
        and config.tokenizer.num_quantizers != config.rvq_baseline.num_quantizers
    ):
        raise ValueError(
            f"{source}: tokenizer.num_quantizers ({config.tokenizer.num_quantizers}) must equal "
            f"rvq_baseline.num_quantizers ({config.rvq_baseline.num_quantizers}) -- the generator's "
            f"num_codebooks is resolved from tokenizer.num_quantizers, while the RVQ tokenizer model "
            f"itself is built from rvq_baseline.num_quantizers. A silent mismatch shapes the "
            f"generator's token embedding table wrong for the tokenizer it will actually decode."
        )
