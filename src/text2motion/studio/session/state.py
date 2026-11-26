from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from text2motion.studio.contracts import (
    MotionInferenceClientPort,
    StudioLaunchArguments,
)
from text2motion.studio.scene import StudioSceneNodes
from text2motion.studio.session.buffers import StudioStreamBuffers
from text2motion.studio.session.debounce import LiveGenerationDebouncer
from text2motion.studio.session.fit_state import SmplxFitState
from text2motion.studio.session.registry import (
    StudioModelEntry,
    load_model_registry,
    resolve_step_budget,
)
from text2motion.studio.studio_config import StudioConfig

logger = logging.getLogger(__name__)


############################################################
#       StudioSession
############################################################
class StudioSession:
    def __init__(
        self,
        client: MotionInferenceClientPort,
        model_dir: str,
        device: str,
        args: StudioLaunchArguments,
        scene,
        config: StudioConfig,
    ) -> None:
        self.client = client
        self.config = config
        self.device = device
        self.model_dir = model_dir
        self.apply_model_limits(client.hello)

        self.apply_sampling_settings(args)
        self.apply_display_settings(config)
        self.apply_fit_defaults(model_dir, config)
        self.load_launch_models(args, config)
        self.start_prompt_state()
        self.build_runtime_parts(scene)
        self.follow_cam = True
        self.styled = False

    def apply_sampling_settings(self, args: StudioLaunchArguments) -> None:
        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.cfg_scale = float(args.cfg_scale)
        self.steps = resolve_step_budget(args, self.downsample, self.max_steps)

    def apply_display_settings(self, config: StudioConfig) -> None:
        self.window_frames = int(config.display.window_frames)
        self.fit_lookahead_chunks = int(config.display.fit_lookahead_chunks)
        self.stream_lead_frames = int(config.display.stream_lead_frames)
        self.upsample = int(config.display.upsample)
        if self.upsample < 1:
            raise ValueError(f"display.upsample must be >= 1, got {self.upsample}")
        self.window_steps = max(8, self.window_frames // self.downsample)

    def apply_fit_defaults(self, model_dir: str, config: StudioConfig) -> None:
        self.gender_idx = 0
        self.fit_body = bool(model_dir)
        self.show_skin = True
        self.skin_color = config.appearance.skin_color
        self.user_betas = np.zeros(config.fit.num_betas, dtype=np.float32)

    def load_launch_models(self, args: StudioLaunchArguments, config: StudioConfig) -> None:
        self.backbone = args.backbone
        launch_entry = StudioModelEntry(
            name=Path(str(args.ckpt)).stem,
            version="local",
            config=str(args.config),
            ckpt=str(args.ckpt),
            tokenizer_ckpt=str(args.tokenizer_ckpt),
            backbone=str(args.backbone),
        )
        self.launch_entry = launch_entry
        self.models, self.model_idx = load_model_registry(launch_entry, config.model_registry)
        self.loading_model = False
        self.load_status = f"active: {self.models[self.model_idx].label}"

    def start_prompt_state(self) -> None:
        self.prompt_text = ""
        self.status = "ready -- describe a motion below and press Enter"
        self.generating = False
        self.history: list[str] = []

    def build_runtime_parts(self, scene) -> None:
        self.live = LiveGenerationDebouncer()
        self.live.signature = (self.temperature, self.top_p, self.steps)
        self.stream_buffers = StudioStreamBuffers()
        self.smplx_fit = SmplxFitState()
        self.nodes = StudioSceneNodes(scene)

    def apply_model_limits(self, hello: dict) -> None:
        def default_step_budget(trained_steps: int, max_steps: int) -> int:
            return min(trained_steps, max_steps) if max_steps else trained_steps

        self.downsample = int(hello["downsample"])
        self.trained_steps = int(hello["trained_steps"])
        self.max_steps = int(hello["max_steps"])
        self.default_steps = default_step_budget(self.trained_steps, self.max_steps)

    def apply_config(self, config: StudioConfig) -> str:
        previous = len(self.models)
        self.config = config
        self.window_frames = int(config.display.window_frames)
        self.window_steps = max(8, self.window_frames // self.downsample)
        self.fit_lookahead_chunks = int(config.display.fit_lookahead_chunks)
        self.stream_lead_frames = int(config.display.stream_lead_frames)
        self.skin_color = config.appearance.skin_color
        self.models, self.model_idx = load_model_registry(
            self.models[self.model_idx], config.model_registry
        )
        added = len(self.models) - previous
        return (
            f"settings reloaded -- "
            f"window {self.window_frames} frames, {len(self.models)} models"
            + (f" (+{added})" if added > 0 else "")
        )

    @property
    def gender(self) -> str:
        return self.config.genders[self.gender_idx]
