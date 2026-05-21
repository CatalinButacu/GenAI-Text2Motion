from __future__ import annotations

import logging
import time
from typing import Any

from src.modules import motion, planner, render, understanding
from src.shared.config import PipelineConfig

log = logging.getLogger(__name__)


class Pipeline:
    """Thin orchestrator. All stage logic lives inside each module's invoke()."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        log.info("Pipeline ready (device=%s)", self.config.device)

    def run(self, prompt: str, output_name: str = "output") -> dict[str, Any]:
        prompt = (prompt or "").strip()[: self.config.prompt_max_chars]

        if not prompt:
            return {"prompt": "", "error": "empty prompt"}

        t0 = time.time()
        cfg = self.config

        parsed = understanding.invoke(prompt, cfg.understanding)
        planned = planner.invoke(parsed, cfg.planner)
        clips = motion.invoke(planned, cfg.motion)
        video = render.invoke(clips, cfg.video_path(output_name), cfg.render)

        return {
            "prompt": prompt,
            "parsed_scene": parsed,
            "planned_scene": planned,
            "motion_clips": clips,
            "video_path": video,
            "elapsed_seconds": time.time() - t0,
        }
