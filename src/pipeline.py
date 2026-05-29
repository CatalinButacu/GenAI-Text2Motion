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

    def run(
        self, prompt: str, output_name: str = "output", stream: bool = False, viewer: bool = False
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()[: self.config.prompt_max_chars]

        if not prompt:
            return {"prompt": "", "error": "empty prompt"}

        t0 = time.time()
        cfg = self.config

        def tick(label: str, t_start: float) -> float:
            elapsed = time.time() - t_start
            if stream:
                print(f"  [{elapsed:5.1f}s]  {label}", flush=True)
            log.info("%s  (%.2fs)", label, elapsed)
            return time.time()

        if stream:
            print(f"\n>>> PIPELINE START: {prompt!r}", flush=True)

        t = time.time()
        parsed = understanding.invoke(prompt, cfg.understanding)
        if stream:
            print("\n[1/4] Understanding", flush=True)
            print(f"       entities : {[e.name for e in parsed.entities]}", flush=True)
            print(f"       actions  : {[a.action_type for a in parsed.actions]}", flush=True)
        t = tick("understanding done", t)

        planned = planner.invoke(parsed, cfg.planner)
        if stream:
            print("\n[2/4] Planner", flush=True)
            for a in planned.actions:
                dur = getattr(a, "duration", None)
                dur_str = f"{dur:.1f}s" if dur is not None else "?"
                print(f"       -> {a.action_type}  dur={dur_str}", flush=True)
        t = tick("planner done", t)

        clips = motion.invoke(planned, cfg.motion)
        if stream:
            print(f"\n[3/4] Motion SSM  ({len(clips)} clip(s))", flush=True)
            for actor, c in clips.items():
                frames = len(c.smplx_params) if c.smplx_params is not None else "?"
                print(f"       {actor!r}: action={c.action!r}  frames={frames}", flush=True)
        t = tick("motion done", t)

        if viewer:
            if stream:
                print("\n[4/4] Viewer  (interactive — close window to exit)", flush=True)
            render.view_interactive(clips, cfg.render)
            video = ""
        else:
            video = render.invoke(clips, cfg.video_path(output_name), cfg.render)
            if stream:
                print(f"\n[4/4] Render -> {video}", flush=True)
        tick("render done", t)

        total = time.time() - t0
        if stream:
            print(f"\n>>> DONE  total={total:.1f}s\n", flush=True)

        return {
            "prompt": prompt,
            "parsed_scene": parsed,
            "planned_scene": planned,
            "motion_clips": clips,
            "video_path": video,
            "elapsed_seconds": total,
        }
