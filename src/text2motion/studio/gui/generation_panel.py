from __future__ import annotations

from pathlib import Path

import imgui

from text2motion.studio.contracts import ViewerHost
from text2motion.studio.gui.avatar_section import AvatarSection
from text2motion.studio.gui.layout import WINDOW_FLAGS, PanelLayout
from text2motion.studio.gui.sampling import SamplingSection
from text2motion.studio.live.actions import log_action
from text2motion.studio.live.controller import StudioGenerationController


############################################################
#       GenerationPanel
############################################################
class GenerationPanel:
    def __init__(
        self,
        host: ViewerHost,
        state,
        generation: StudioGenerationController,
        layout: PanelLayout,
        sampling: SamplingSection,
        avatar_section: AvatarSection,
    ) -> None:
        self.host = host
        self.state = state
        self.generation = generation
        self.layout = layout
        self.sampling = sampling
        self.avatar_section = avatar_section

    def gui_params(self) -> None:
        *_, params, _, _ = self.layout.rects()
        self.layout.dock(params)
        imgui.begin("Generation", False, WINDOW_FLAGS)
        self.model_controls()
        self.sampling.sampling_controls()
        self.avatar_section.avatar_controls()
        self.prompt_history()
        imgui.end()

    def model_controls(self) -> None:
        self.layout.section("model")
        picked, self.state.model_idx = imgui.combo(
            "model", self.state.model_idx, [e.label for e in self.state.models]
        )
        if picked:
            log_action("select model", self.state.models[self.state.model_idx].key)
        entry = self.state.models[self.state.model_idx]
        self.layout.dim_text(
            f"pinned from training: {entry.backbone} backbone, tokenizer"
            f" {Path(entry.tokenizer_ckpt).stem}, config {Path(entry.config).stem}"
        )
        busy = self.state.generating or self.state.loading_model
        if imgui.button("Load model", width=-1) and not busy:
            log_action("press Load model", self.state.models[self.state.model_idx].key)
            self.generation.loader.start_model_load()
        if imgui.button("Reload settings + models (R)", width=-1) and not busy:
            log_action("press Reload settings")
            self.host.reload_settings()
        self.layout.dim_text(self.state.load_status)

    def prompt_history(self) -> None:
        if len(self.state.history) > 0 and imgui.collapsing_header("prompt history")[0]:
            for i, past in enumerate(reversed(self.state.history[-10:])):
                clicked, _ = imgui.selectable(f"{past[:44]}##hist{i}")
                if clicked:
                    self.state.prompt_text = past
