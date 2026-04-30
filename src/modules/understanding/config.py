from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpacyConfig:
    model: str = "en_core_web_sm"


@dataclass
class ParserConfig:
    spacy: SpacyConfig = field(default_factory=SpacyConfig)
