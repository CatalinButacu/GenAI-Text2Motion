from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.modules.motion.config import MotionConfig
from src.modules.planner.config import PlannerConfig
from src.modules.render.config import RenderConfig
from src.modules.understanding.config import ParserConfig


@dataclass
class PipelineConfig:
    outputDir: str = "outputs"
    duration: float = 5.0
    fps: int = 30
    device: str = "cuda"
    promptMaxChars: int = 2000

    understanding: ParserConfig = field(default_factory=ParserConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def __post_init__(self) -> None:
        self.planner.baseDuration = self.duration
        self.render.fps = self.fps

    def videoPath(self, outputName: str) -> str:
        return os.path.join(self.outputDir, "videos", f"{outputName}.mp4")
