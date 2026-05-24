"""Interactive aitviewer subclass with a chat input bar at the bottom-center.

Lets the user type prompts inside the 3D window and re-runs the text-to-motion
pipeline on demand without leaving the viewer. The pipeline is invoked on a
background thread so the OpenGL render loop stays responsive while the SSM
generates new motion.

Layout (1280×800 default):
  ┌──────────┬──────────────────────────────────────────┐
  │ Editor   │                                          │
  │  260×72% │          3-D viewport                    │
  │          │                                          │
  ├──────────┤                                          │
  │Playback  │                                          │
  │  260×17% ├──────────────────────────────────────────┤
  │          │   Text-to-Motion Chat  (bottom-center)   │
  └──────────┴──────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import threading

import imgui
import numpy as np
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

from .smplx_render import smplx_params2_sequence

log = logging.getLogger(__name__)

MAX_HISTORY = 8
MAX_INPUT_LEN = 256

# Left panel column width (pixels)
_LEFT_W = 270
# Chat panel height (pixels)
_CHAT_H = 200


class ChatViewer(Viewer):
    """Viewer with a chat bar that drives the text-to-motion pipeline live."""

    def __init__(
        self,
        pipeline_runner,
        fps: int = 30,
        title: str = "Text-to-Motion  |  GenAI Text2Motion",
        size: tuple[int, int] = (1280, 800),
    ) -> None:
        super().__init__(title=title, size=size)

        self.pipeline_runner = pipeline_runner
        self.fps = fps
        self.input_buffer = ""
        self.history: list[tuple[str, str]] = []
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

        self.gui_controls["chat"] = self.gui_chat

    # ------------------------------------------------------------------
    # Layout overrides — keep left column tidy, chat at bottom-center
    # ------------------------------------------------------------------

    def gui_scene(self) -> None:
        """Editor panel: left column, top."""
        h = self.window_size[1]
        editor_h = h - 170  # leaves room for Playback below
        imgui.set_next_window_position(10, 10, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(_LEFT_W, editor_h, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Editor", None)
        if expanded:
            self.scene.gui_editor(imgui, self.viewports, self.viewport_mode)
        imgui.end()

    def gui_playback(self) -> None:
        """Playback panel: left column, directly below Editor — fully overridden to
        prevent aitviewer's parent from resetting the position/size."""
        h = self.window_size[1]
        editor_h = h - 170
        imgui.set_next_window_position(10, editor_h + 15, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(_LEFT_W, 155, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Playback", None)
        if expanded:
            u, run_anim = imgui.checkbox(
                "Run animations [{}]".format(self._shortcut_names[self._pause_key]),
                self.run_animations,
            )
            if u:
                self.toggle_animation(run_anim)

            from array import array as _array
            import numpy as _np
            frametime_avg = _np.mean(self._past_frametimes[self._past_frametimes > 0.0])
            fps_avg = 1 / frametime_avg
            ms_avg = frametime_avg * 1000.0
            ms_last = self._past_frametimes[-1] * 1000.0
            imgui.plot_lines(
                "Internal {:.1f} fps @ {:.2f} ms [{:.2f}ms]".format(fps_avg, ms_avg, ms_last),
                _array("f", (1.0 / self._past_frametimes).tolist()),
                scale_min=0, scale_max=100.0, graph_size=(_LEFT_W - 20, 20),
            )
            _, self.playback_fps = imgui.drag_float(
                "Playback fps", self.playback_fps, 0.1,
                min_value=1.0, max_value=120.0, format="%.1f",
            )
            imgui.same_line(spacing=10)
            imgui.text(f"({self.playback_fps / self.scene.fps:.2f}x)")

            n_frames = self.scene.n_frames
            _, self.scene.current_frame_id = imgui.slider_int(
                "Frame##seq", self.scene.current_frame_id, 0, n_frames - 1,
            )
            self.prevent_background_interactions()
        if imgui.collapsing_header("Advanced options")[0]:
            _, self.playback_without_skipping = imgui.checkbox(
                "Playback without skipping", self.playback_without_skipping
            )
        imgui.end()

    def gui_chat(self) -> None:
        """Chat panel: bottom-right of viewport, starts where left column ends."""
        w, h = self.window_size
        chat_w = w - _LEFT_W - 20
        x = _LEFT_W + 10
        y = h - _CHAT_H - 10

        imgui.set_next_window_position(x, y, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(chat_w, _CHAT_H, imgui.FIRST_USE_EVER)
        imgui.set_next_window_bg_alpha(0.88)

        flags = imgui.WINDOW_NO_COLLAPSE
        opened, _ = imgui.begin("Text-to-Motion Chat", None, flags)
        if not opened:
            imgui.end()
            return

        # History strip
        hist_h = _CHAT_H - 68
        imgui.begin_child("##chat-hist", height=hist_h, border=False)
        for prompt, status in self.history:
            color = (
                (0.5, 1.0, 0.5, 1.0) if status == "ok"
                else (1.0, 0.85, 0.35, 1.0) if status == "running"
                else (1.0, 0.45, 0.45, 1.0)
            )
            imgui.text_colored(f"▶ {prompt}", *color)
            if status not in ("ok", "running"):
                imgui.same_line()
                imgui.text_disabled(f"  [{status}]")
        if self.busy and self.busy_msg:
            imgui.text_colored(f"⏳ {self.busy_msg}", 1.0, 0.85, 0.35, 1.0)
        imgui.set_scroll_here_y(1.0)
        imgui.end_child()

        imgui.separator()

        # Input + Send on one row
        input_w = chat_w - 105
        imgui.set_next_item_width(input_w)
        enter_pressed, self.input_buffer = imgui.input_text(
            "##prompt", self.input_buffer, MAX_INPUT_LEN,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        imgui.same_line()
        if self.busy:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            imgui.button("Generate", width=85)
            imgui.pop_style_var()
            send = False
        else:
            send = imgui.button("Generate", width=85)

        if (enter_pressed or send) and self.input_buffer.strip() and not self.busy:
            self.submit_prompt(self.input_buffer)
            self.input_buffer = ""

        imgui.end()

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

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
        """Kick off the pipeline on a background thread."""
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
            except Exception as exc:  # noqa: BLE001
                log.exception("[chat] pipeline failed")
                status = f"error: {exc}"
            with self.lock:
                self.pending_clip = clip_dict
                if self.history and self.history[-1][0] == prompt:
                    self.history[-1] = (prompt, status)
                self.busy = False
                self.busy_msg = ""

        threading.Thread(target=worker, daemon=True).start()

    def on_update(self) -> None:
        with self.lock:
            clip = self.pending_clip
            self.pending_clip = None
        if clip is not None:
            self.swap_clip(clip)

    def gui(self) -> None:
        self.on_update()
        super().gui()

