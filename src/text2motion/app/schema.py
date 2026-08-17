from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


def coerce(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(annotation)

    if origin in (Union, types.UnionType):
        candidates = [arg for arg in get_args(annotation) if arg is not type(None)]
        return coerce(candidates[0], value)

    if origin in (tuple, Sequence):
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(coerce(args[0], item) for item in value)
        return tuple(coerce(arg, item) for arg, item in zip(args, value, strict=True))

    if origin is list:
        (arg,) = get_args(annotation)
        return [coerce(arg, item) for item in value]

    if origin is dict:
        key_type, value_type = get_args(annotation)
        return {coerce(key_type, k): coerce(value_type, v) for k, v in value.items()}

    if is_dataclass(annotation):
        return from_mapping(annotation, value)

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return annotation(value)
        if issubclass(annotation, Path):
            return Path(value)
        if annotation in (bool, int, float, str):
            return annotation(value)

    return value


def from_mapping(cls: type[T], data: Mapping[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")

    hints = get_type_hints(cls)
    known = {field.name for field in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown key(s) {sorted(unknown)}. Known keys: {sorted(known)}. "
            f"A silently ignored key is a setting you think is applied but is not."
        )

    return cls(**{name: coerce(hints[name], value) for name, value in data.items()})


def load_dataclass(cls: type[T], path: str | Path) -> T:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_mapping(cls, raw)
