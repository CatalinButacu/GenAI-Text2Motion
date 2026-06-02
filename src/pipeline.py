from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from functools import cached_property
from typing import Any

import numpy as np
import torch

from src.modules.motion.generator import MotionGenerator
from src.modules.motion.models import MotionClip
from src.modules.planner.planner import ScenePlanner
from src.modules.render import render_clip_to_file, view_clip
from src.modules.render.chat_viewer import ChatViewer
from src.modules.runtime import StreamBus, StreamCapabilityError, StreamMetricsCollector
from src.modules.understanding.spacy import SpacyParser
from src.shared.config import PipelineConfig

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        log.info("Pipeline ready (device=%s, seed=%d)", self.config.device, self.config.seed)

    @cached_property
    def parser(self) -> SpacyParser:
        return SpacyParser(self.config.understanding)

    @cached_property
    def layout(self) -> ScenePlanner:
        return ScenePlanner(self.config.planner)

    @cached_property
    def motion(self) -> MotionGenerator:
        return MotionGenerator(self.config.motion)

    def parse_and_plan(self, prompt: str) -> dict[str, Any] | None:
        prompt = (prompt or "").strip()[: self.config.prompt_max_chars]

        if not prompt:
            return None

        parsed = self.parser.parse(prompt)
        planned = self.layout.plan(parsed)
        return {"prompt": prompt, "parsed_scene": parsed, "planned_scene": planned}

    def generate(self, prompt: str) -> dict[str, Any] | None:
        base = self.parse_and_plan(prompt)

        if base is None:
            return None

        clips = self.motion.generate_for_scene(base["planned_scene"])
        return {**base, "motion_clips": clips}

    def generate_stream(self, prompt: str) -> Iterator[np.ndarray]:
        base = self.parse_and_plan(prompt)

        if base is None:
            return

        yield from self.motion.stream_for_scene(base["planned_scene"])

    def validate_streaming(
        self,
        prompt: str,
        max_stream_chunks: int | None = None,
        bus_size: int = 32,
    ) -> dict[str, float | int | None]:
        if max_stream_chunks is not None and max_stream_chunks <= 0:
            raise ValueError("max_stream_chunks must be positive")

        self.motion.backend.require_stream_capable()
        bus = StreamBus(max_size=bus_size)
        metrics = StreamMetricsCollector()

        for chunk in self.generate_stream(prompt):
            bus.push(chunk)
            metrics.observe_produced()
            packet = bus.pop()

            if packet is not None:
                metrics.observe_consumed(packet)

            if max_stream_chunks is not None and metrics.consumed_chunks >= max_stream_chunks:
                break

        summary = metrics.summary(dropped_chunks=bus.dropped_chunks)
        return {
            "produced_chunks": summary.produced_chunks,
            "consumed_chunks": summary.consumed_chunks,
            "dropped_chunks": summary.dropped_chunks,
            "bad_chunks": summary.bad_chunks,
            "first_chunk_latency_ms": summary.first_chunk_latency_ms,
            "inter_chunk_p50_ms": summary.inter_chunk_p50_ms,
            "inter_chunk_p95_ms": summary.inter_chunk_p95_ms,
            "wall_time_ms": summary.wall_time_ms,
        }

    def render_to_file(self, prompt: str, output_name: str = "output") -> dict[str, Any] | None:
        result = self.generate(prompt)

        if result is None:
            return None

        clip = next(iter(result["motion_clips"].values()))
        path = self.config.video_path(output_name)

        t = time.time()
        render_clip_to_file(clip, path, self.config.render)
        log.info("render done (%.2fs)", time.time() - t)

        result["video_path"] = path
        return result

    def run_viewer(self, prompt: str) -> dict[str, Any] | None:
        result = self.generate(prompt)

        if result is None:
            return None

        view_clip(next(iter(result["motion_clips"].values())), self.config.render)
        return result

    def run_chat(self) -> None:
        log.info("chat window open — type prompts in the chat bar")
        ChatViewer(
            pipeline_runner=self.chat_step,
            stream_pipeline_runner=self.generate_stream,
            stream_coord_system=self.motion.backend.coord_system,
            render_cfg=self.config.render,
            fps=self.config.fps,
        ).run()

    def chat_step(self, prompt: str) -> MotionClip | None:
        result = self.generate(prompt)

        if result is None or not result["motion_clips"]:
            return None

        return next(iter(result["motion_clips"].values()))
