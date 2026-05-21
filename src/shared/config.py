from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.modules.motion.config import MotionConfig
from src.modules.planner.config import PlannerConfig
from src.modules.render.config import RenderConfig
from src.modules.understanding.config import ParserConfig


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    duration: float = 5.0
    fps: int = 30
    device: str = "cuda"
    prompt_max_chars: int = 2000

    understanding: ParserConfig = field(default_factory=ParserConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def __post_init__(self) -> None:
        self.planner.base_duration = self.duration
        self.render.fps = self.fps

    def video_path(self, output_name: str) -> str:
        return os.path.join(self.output_dir, "videos", f"{output_name}.mp4")
