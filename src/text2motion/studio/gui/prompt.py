from __future__ import annotations

import time

import imgui

from text2motion.studio.generation import (
    StudioGenerationController,
    log_action,
    log_toggle,
)
from text2motion.studio.gui.layout import WINDOW_FLAGS, PanelLayout


############################################################
#       PromptPanel
############################################################
class PromptPanel:
    def __init__(
        self,
        state,
        generation: StudioGenerationController,
        layout: PanelLayout,
    ) -> None:
        self.state = state
        self.generation = generation
        self.layout = layout

    def gui_prompt(self) -> None:
        *_, prompt = self.layout.rects()
        self.layout.dock(prompt)
        flags = WINDOW_FLAGS | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR
        imgui.begin("##prompt_bar", False, flags)

        button_w = 96.0
        enter = self.prompt_input(button_w)
        imgui.same_line()
        self.generate_button(button_w, enter)
        self.prompt_toggles()

        if self.state.generating:
            self.progress_bars()
        self.layout.dim_text(self.state.status)
        imgui.end()

    def prompt_input(self, button_w: float) -> bool:
        imgui.push_item_width(imgui.get_content_region_available()[0] - button_w - 8)
        enter, self.state.prompt_text = imgui.input_text_with_hint(
            "##prompt",
            'describe a motion, e.g. "a person walks in a circle" -- Enter to generate',
            self.state.prompt_text,
            256,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        if self.state.prompt_text != self.state.live.prev_prompt:
            self.state.live.mark_dirty(self.state.prompt_text, time.perf_counter())
        imgui.pop_item_width()
        return enter

    def prompt_toggles(self) -> None:
        changed, self.state.live.enabled = imgui.checkbox(
            "live (regenerate as you type)", self.state.live.enabled
        )
        if changed:
            log_toggle("live regenerate", self.state.live.enabled)
        imgui.same_line()
        label = "hide skin (H)" if self.state.show_skin else "show skin (H)"
        if imgui.button(label):
            self.state.show_skin = not self.state.show_skin
            log_toggle("skin", self.state.show_skin)

    def generate_button(self, width: float, enter: bool) -> None:
        if not self.state.generating or self.state.live.enabled:
            if imgui.button("Generate", width=width) or enter:
                self.generation.submit_prompt()
            return
        if imgui.button("Stop", width=width):
            log_action("press Stop", f"at {self.state.stream_buffers.frames_emitted} frames")
            self.generation.stop()

    def progress_bars(self) -> None:
        buffers = self.state.stream_buffers
        avail = imgui.get_content_region_available()[0]
        done = buffers.frames_emitted
        total = self.state.steps * self.state.downsample * self.state.upsample
        label = f"motion {done}/{total} frames" if total else f"motion {done} frames (forever)"
        bar = buffers.gen_progress if total else (done % 100) / 100.0
        if not buffers.current_fit_body:
            imgui.progress_bar(bar, (-1, 15), label)
            return
        imgui.progress_bar(bar, (avail * 0.55, 15), label)
        imgui.same_line()
        fit = buffers.fit_progress or 0.0
        imgui.progress_bar(fit, (-1, 15), f"body fit {int(fit * 100)}%")
