from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from text2motion.dataset.contracts import SOURCE_NAMES
from text2motion.dataset.sources.amass.source import AmassSource
from text2motion.dataset.sources.base import RecordingSource
from text2motion.dataset.sources.humanml3d.source import HumanML3DSource

SOURCES: Mapping[str, type[RecordingSource]] = MappingProxyType(
    {AmassSource.name: AmassSource, HumanML3DSource.name: HumanML3DSource}
)


def source_named(name: str) -> type[RecordingSource]:
    if name not in SOURCES:
        raise KeyError(f"unknown motion source {name!r}; registered: {sorted(SOURCES)}")
    return SOURCES[name]


if tuple(sorted(SOURCES)) != tuple(sorted(SOURCE_NAMES)):
    raise RuntimeError("registry and SOURCE_NAMES in contracts.py disagree")
