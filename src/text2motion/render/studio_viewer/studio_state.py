from __future__ import annotations

import argparse
from pathlib import Path

import imgui
import numpy as np

from text2motion.render.studio_viewer.constants import GENDERS, N_BETAS, SKIN_COLOR
from text2motion.render.studio_viewer.registry import ModelEntry, load_model_registry
from text2motion.render.studio_viewer.scene_nodes import (
    FitState,
    LiveRegen,
    SceneNodes,
    StreamBuffers,
)


class StudioState:
    def __init__(
        self, client, model_dir: str, device: str, args: argparse.Namespace, scene
    ) -> None:
        self.client = client
        self.device = device
        self.model_dir = model_dir
        self.downsample = int(client.hello["downsample"])
        self.max_steps = int(client.hello["max_steps"])

        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.cfg_scale = float(args.cfg_scale)
        self.steps = min(int(args.steps), self.max_steps)

        self.gender_idx = 0
        self.fit_body = bool(model_dir)
        self.skin_color = SKIN_COLOR
        self.user_betas = np.zeros(N_BETAS, dtype=np.float32)

        self.backbone = args.backbone
        launch_entry = ModelEntry(
            label=f"launch args: {Path(args.ckpt).stem}",
            config=args.config,
            ckpt=args.ckpt,
            tokenizer_ckpt=args.tokenizer_ckpt,
            backbone=args.backbone,
        )
        self.models, self.model_idx = load_model_registry(launch_entry)
        self.loading_model = False
        self.load_status = f"active: {self.models[self.model_idx].label}"

        self.prompt_text = ""
        self.status = "ready -- describe a motion below and press Enter"
        self.generating = False
        self.history: list[str] = []

        self.live = LiveRegen()
        self.live.signature = (self.temperature, self.top_p, self.steps)
        self.buf = StreamBuffers()
        self.fit = FitState()
        self.nodes = SceneNodes(scene)
        self.follow_cam = True

        self.flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE
        self.styled = False

    @property
    def gender(self) -> str:
        return GENDERS[self.gender_idx]
