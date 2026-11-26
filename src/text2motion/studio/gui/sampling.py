from __future__ import annotations

import imgui

from text2motion.studio.contracts import ViewerHost
from text2motion.studio.gui.layout import PanelLayout
from text2motion.studio.live.actions import log_toggle


############################################################
#       SamplingSection
############################################################
class SamplingSection:
    def __init__(
        self,
        host: ViewerHost,
        state,
        layout: PanelLayout,
    ) -> None:
        self.host = host
        self.state = state
        self.layout = layout

    def sampling_controls(self) -> None:
        self.layout.section("sampling")
        _, self.state.temperature = imgui.slider_float(
            "temperature", self.state.temperature, 0.1, 2.0
        )
        _, self.state.top_p = imgui.slider_float("top_p", self.state.top_p, 0.1, 1.0)
        _, self.state.cfg_scale = imgui.slider_float("cfg_scale", self.state.cfg_scale, 1.0, 12.0)
        if self.state.max_steps == 0:
            self.forever_toggle()
        if self.state.steps == 0:
            self.layout.dim_text(
                "unbounded: this recurrent backbone streams until you press stop. "
                f"The viewer keeps the last {self.state.window_frames} frames."
            )
        else:
            self.steps_slider()
            self.steps_description()

    def forever_toggle(self) -> None:
        forever = self.state.steps == 0
        clicked, forever = imgui.checkbox("run forever (bounded state)", forever)
        if clicked:
            self.state.steps = 0 if forever else self.state.default_steps
            log_toggle("run forever", forever)

    def steps_slider(self) -> None:
        upper = self.state.max_steps or self.state.window_steps
        _, self.state.steps = imgui.slider_int("steps (tokens)", self.state.steps, 8, upper)

    def steps_description(self) -> None:
        seconds = (
            self.state.steps * self.state.downsample * self.state.upsample / self.host.playback_fps
        )
        cap = (
            f"{self.state.max_steps} is the model's position-table hard cap."
            if self.state.max_steps
            else "this backbone has no position cap -- tick 'run forever' for unbounded."
        )
        trained_seconds = (
            self.state.trained_steps
            * self.state.downsample
            * self.state.upsample
            / self.host.playback_fps
        )
        self.layout.dim_text(
            f"= {self.state.steps * self.state.downsample} frames (~{seconds:.1f}s @"
            f" {self.host.playback_fps:.0f} fps). Trained at {self.state.trained_steps} steps "
            f"(~{trained_seconds:.1f}s); much lower looks clipped; {cap}"
        )
