from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderConfig:
    fps: int = 30
    gender: str = "neutral"
    width: int = 1280
    height: int = 720
