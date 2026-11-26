from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import GenericAlias, MappingProxyType
from typing import Any, TypeVar, cast

import yaml
from pydantic import ConfigDict, TypeAdapter

from text2motion.app.config.schema import ApplicationConfig, MotionServiceConfig
from text2motion.app.config.validation import (
    check_representation_block,
    check_root_keys,
    training_values,
    validate_config,
)

_T = TypeVar("_T")


_STRICT_KEYS = ConfigDict(extra="forbid")


def from_mapping(cls: type[_T], data: Mapping[str, Any]) -> _T:
    single = cast("type[tuple[_T]]", GenericAlias(tuple, (cls,)))
    strict_single = TypeAdapter(single, config=_STRICT_KEYS)
    return strict_single.validate_python((dict(data),))[0]


def load_service_config(path: str | Path) -> MotionServiceConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_mapping(MotionServiceConfig, with_environment("service", {"service": raw}))


def service_policy(service: MotionServiceConfig) -> dict[str, object]:
    return {
        "anchor_every_tokens": service.anchor_every_tokens,
        "anchor_context_tokens": service.anchor_context_tokens,
        "cuda_graph_decode": service.cuda_graph_decode,
    }


def load_dataclass(cls: type[_T], path: str | Path) -> _T:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_mapping(cls, raw)


_ENVIRONMENT_OVERRIDES = MappingProxyType(
    {
        "paths": MappingProxyType(
            {
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
            }
        ),
        "service": MappingProxyType(
            {
                "bind_host": ("TEXT2MOTION_BIND_HOST",),
                "connect_host": ("TEXT2MOTION_CONNECT_HOST", "TEXT2MOTION_SERVICE_HOST"),
                "port": ("TEXT2MOTION_SERVICE_PORT",),
                "idle_seconds": ("TEXT2MOTION_IDLE_SECONDS",),
            }
        ),
    }
)


def with_environment(section: str, raw: dict) -> dict:
    values = dict(raw.get(section, {}))
    for key, names in _ENVIRONMENT_OVERRIDES[section].items():
        value = next((os.getenv(name) for name in names if os.getenv(name)), None)
        if value is not None:
            values[key] = value
    return values


def load_config(path: str | Path) -> ApplicationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    check_root_keys(raw, path)
    check_representation_block(raw, path)
    values = {key: raw[key] for key in ApplicationConfig.__dataclass_fields__ if key in raw}
    values["paths"] = with_environment("paths", raw)
    values["service"] = with_environment("service", raw)
    values["train"] = training_values(raw)
    config = from_mapping(ApplicationConfig, values)
    validate_config(config, path)
    return config
