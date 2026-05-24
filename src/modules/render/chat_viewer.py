"""Interactive aitviewer subclass with a chat input bar at the bottom-center.

Lets the user type prompts inside the 3D window and re-runs the text-to-motion
pipeline on demand without leaving the viewer. The pipeline is invoked on a
background thread so the OpenGL render loop stays responsive while the SSM
generates new motion.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import imgui
import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

from .smplx_render import smplx_params2_sequence

log = logging.getLogger(__name__)

MAX_HISTORY = 8
MAX_INPUT_LEN = 256


class ChatViewer(Viewer):
    """Viewer with a chat bar that drives the text-to-motion pipeline live."""

    def __init__(
        self,
        pipeline_runner,
        initial_clip: dict | None = None,
        fps: int = 30,
        title: str = "Text-to-Motion Chat",
        size: tuple[int, int] = (1280, 800),
    ) -> None:
        super().__init__(title=title, size=size)

        self.pipeline_runner = pipeline_runner
        self.fps = fps
        self.input_buffer = ""
        self.history: list[tuple[str, str]] = []  # (prompt, status)
        self.busy = False
        self.busy_msg = ""
        self.pending_clip: dict | None = None
        self.lock = threading.Lock()

        self.scene.fps = fps
        self.playback_fps = fps

        cam = self.scene.camera
        if cam is not None:
            cam.position = np.array([0.0, 1.5, 4.5])
            cam.target = np.array([0.0, 1.0, 0.0])

        if initial_clip is not None:
            self.swap_clip(initial_clip)

        # Register the chat panel as an extra GUI control.
        self.gui_controls["chat"] = self.gui_chat

    def swap_clip(self, clip: dict) -> None:
        """Replace the current SMPLSequence in the scene with one built from `clip`."""
        for node in list(self.scene.nodes):
            if isinstance(node, SMPLSequence):
                self.scene.remove(node)

        seq = smplx_params2_sequence(
            clip["smplx_params"],
            betas=clip.get("betas"),
            gender=clip.get("gender", "neutral"),
            input_coord_system=clip.get("input_coord_system", "yup"),
        )
        self.scene.add(seq)
        self.scene.current_frame_id = 0
        self.run_animations = True

    def submit_prompt(self, prompt: str) -> None:
        """Kick off the pipeline on a background thread so the UI stays responsive."""
        prompt = prompt.strip()
        if not prompt or self.busy:
            return

        self.busy = True
        self.busy_msg = f"generating: {prompt[:60]}..."
        self.history.append((prompt, "running"))
        self.history[:] = self.history[-MAX_HISTORY:]

        def worker() -> None:
            status = "ok"
            clip_dict = None
            try:
                clip_dict = self.pipeline_runner(prompt)
                if clip_dict is None:
                    status = "no motion"
            except Exception as e:  # noqa: BLE001 — surface any failure into UI
                log.exception("[chat] pipeline failed")
                status = f"error: {e}"

            with self.lock:
                self.pending_clip = clip_dict
                if self.history and self.history[-1][0] == prompt:
                    self.history[-1] = (prompt, status)
                self.busy = False
                self.busy_msg = ""

        threading.Thread(target=worker, daemon=True).start()

    def on_update(self) -> None:
        """Called every frame on the main thread — safe place to mutate the scene."""
        with self.lock:
            clip = self.pending_clip
            self.pending_clip = None

        if clip is not None:
            self.swap_clip(clip)

    def gui(self) -> None:
        """Override the per-frame GUI pass so we can drain pending updates first."""
        self.on_update()
        super().gui()

    def gui_chat(self) -> None:
        """Render the chat bar at the bottom-center of the window."""
        w, h = self.window_size
        panel_w = int(min(720, w * 0.6))
        panel_h = 180
        x = (w - panel_w) // 2
        y = h - panel_h - 30

        imgui.set_next_window_position(x, y, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(panel_w, panel_h, imgui.FIRST_USE_EVER)
        imgui.set_next_window_bg_alpha(0.85)

        flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR
        opened = imgui.begin("##chat-panel", True, flags)
        if not opened:
            imgui.end()
            return

        imgui.text("Text-to-Motion chat")
        imgui.separator()

        # History strip
        imgui.begin_child("##chat-hist", height=panel_h - 90, border=False)
        for prompt, status in self.history:
            color = (
                (0.6, 1.0, 0.6, 1.0)
                if status == "ok"
                else (1.0, 0.85, 0.4, 1.0)
                if status == "running"
                else (1.0, 0.5, 0.5, 1.0)
            )
            imgui.text_colored(f"> {prompt}", *color)
            if status != "ok":
                imgui.same_line()
                imgui.text_disabled(f"  [{status}]")
        if self.busy and self.busy_msg:
            imgui.text_colored(self.busy_msg, 1.0, 0.85, 0.4, 1.0)
        imgui.end_child()

        # Input row
        imgui.set_next_item_width(panel_w - 110)
        flags_in = imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
        changed, self.input_buffer = imgui.input_text(
            "##chat-input", self.input_buffer, MAX_INPUT_LEN, flags_in
        )
        send_clicked = False
        imgui.same_line()
        if self.busy:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            imgui.button("Send", width=80)
            imgui.pop_style_var()
        else:
            send_clicked = imgui.button("Send", width=80)

        if (changed or send_clicked) and self.input_buffer.strip() and not self.busy:
            prompt = self.input_buffer
            self.input_buffer = ""
            self.submit_prompt(prompt)

        imgui.end()
