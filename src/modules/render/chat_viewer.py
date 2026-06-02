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
from array import array
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import imgui
import numpy as np
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

from src.modules.motion.models import MotionClip

from .config import RenderConfig
from .smplx_render import smplx_params2_sequence

log = logging.getLogger(__name__)

MAX_HISTORY = 8
MAX_INPUT_LEN = 256

# Left panel column width (pixels)

LEFT_W = 270

# aitviewer main menu bar height (pixels)

MENU_H = 22

# Playback panel height (left column, bottom)

PLAYBACK_H = 158

# Chat panel height — 50px taller than Playback

CHAT_H = 208

# RGBA colours used in the history strip for each status value
HISTORY_STATUS_COLORS: dict[str, tuple[float, float, float, float]] = {
    "ok": (0.5, 1.0, 0.5, 1.0),
    "running": (1.0, 0.85, 0.35, 1.0),
}
HISTORY_ERROR_COLOR: tuple[float, float, float, float] = (1.0, 0.45, 0.45, 1.0)


class ChatViewer(Viewer):
    """Viewer with a chat bar that drives the text-to-motion pipeline live."""

    if TYPE_CHECKING:
        # aitviewer's Viewer.__init__ sets self.scene = None before constructing
        # the real Scene, so Pyright infers type None. Override it to Any so that
        # scene.fps / scene.nodes / scene.camera etc. are accepted without stubs.
        scene: Any

    def __init__(
        self,
        pipeline_runner,
        render_cfg: RenderConfig,
        fps: int = 30,
        title: str = "Text-to-Motion  |  GenAI Text2Motion",
        size: tuple[int, int] = (1280, 800),
        stream_pipeline_runner: Callable[[str], Iterator[np.ndarray]] | None = None,
        stream_coord_system: str = "yup",
    ) -> None:
        super().__init__(title=title, size=size)

        self.pipeline_runner = pipeline_runner
        self.stream_pipeline_runner = stream_pipeline_runner
        self.stream_coord_system = stream_coord_system
        self.render_cfg = render_cfg
        self.fps = fps
        self.input_buffer = ""
        self.history: list[tuple[str, str]] = []
        self.busy = False
        self.busy_msg = ""
        self.pending_clip: MotionClip | None = None
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
        """Editor panel: left column, top — starts below menu bar."""
        h = self.window_size[1]
        editor_h = h - MENU_H - CHAT_H - 10  # gap between editor and playback
        imgui.set_next_window_position(10, MENU_H + 3, imgui.ALWAYS)
        imgui.set_next_window_size(LEFT_W, editor_h, imgui.ALWAYS)
        expanded, _ = imgui.begin("Editor", None)
        if expanded:
            self.scene.gui_editor(imgui, self.viewports, self.viewport_mode)
        imgui.end()

    def gui_playback(self) -> None:
        """Playback panel: left column, bottom — top/bottom aligned with Chat."""
        h = self.window_size[1]
        y = h - CHAT_H - 5  # same top edge as Chat
        imgui.set_next_window_position(10, y, imgui.ALWAYS)
        imgui.set_next_window_size(LEFT_W, CHAT_H, imgui.ALWAYS)
        expanded, _ = imgui.begin("Playback", None)
        if expanded:
            u, run_anim = imgui.checkbox(
                f"Run animations [{self._shortcut_names[self._pause_key]}]",
                self.run_animations,
            )
            if u:
                self.toggle_animation(run_anim)

            frametime_avg = np.mean(self._past_frametimes[self._past_frametimes > 0.0])
            fps_avg = 1 / frametime_avg
            ms_avg = frametime_avg * 1000.0
            ms_last = self._past_frametimes[-1] * 1000.0
            imgui.plot_lines(
                f"Internal {fps_avg:.1f} fps @ {ms_avg:.2f} ms [{ms_last:.2f}ms]",
                array("f", (1.0 / self._past_frametimes).tolist()),
                scale_min=0,
                scale_max=100.0,
                graph_size=(LEFT_W - 20, 20),
            )
            _, self.playback_fps = imgui.drag_float(
                "Playback fps",
                self.playback_fps,
                0.1,
                min_value=1.0,
                max_value=120.0,
                format="%.1f",
            )
            imgui.same_line(spacing=10)
            imgui.text(f"({self.playback_fps / self.scene.fps:.2f}x)")

            n_frames = self.scene.n_frames
            _, self.scene.current_frame_id = imgui.slider_int(
                "Frame##seq",
                self.scene.current_frame_id,
                0,
                n_frames - 1,
            )
            self.prevent_background_interactions()
        if imgui.collapsing_header("Advanced options")[0]:
            _, self.playback_without_skipping = imgui.checkbox(
                "Playback without skipping", self.playback_without_skipping
            )
        imgui.end()

    def gui_chat(self) -> None:
        """Chat panel: bottom-right, same y/height as Playback so bottoms align."""
        w, h = self.window_size
        chat_w = w - LEFT_W - 20
        x = LEFT_W + 10
        y = h - CHAT_H - 5

        imgui.set_next_window_position(x, y, imgui.ALWAYS)
        imgui.set_next_window_size(chat_w, CHAT_H, imgui.ALWAYS)
        imgui.set_next_window_bg_alpha(0.88)

        opened, _ = imgui.begin("Text-to-Motion Chat", None, imgui.WINDOW_NO_COLLAPSE)
        if not opened:
            imgui.end()
            return

        # History strip — shrink a bit to fit export button row
        hist_h = CHAT_H - 72
        imgui.begin_child("##chat-hist", height=hist_h, border=False)
        for prompt, status in self.history:
            color = HISTORY_STATUS_COLORS.get(status, HISTORY_ERROR_COLOR)
            imgui.text_colored(f"▶ {prompt}", *color)
            if status not in ("ok", "running"):
                imgui.same_line()
                imgui.text_disabled(f"  [{status}]")
        if self.busy and self.busy_msg:
            imgui.text_colored(f"⏳ {self.busy_msg}", 1.0, 0.85, 0.35, 1.0)
        imgui.set_scroll_here_y(1.0)
        imgui.end_child()

        imgui.separator()

        # Row 1: prompt input + Generate
        input_w = chat_w - 110
        imgui.set_next_item_width(input_w)
        enter_pressed, self.input_buffer = imgui.input_text(
            "##prompt",
            self.input_buffer,
            MAX_INPUT_LEN,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        imgui.same_line()
        if self.busy:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            imgui.button("Generate", width=95)
            imgui.pop_style_var()
            send = False
        else:
            send = imgui.button("Generate", width=95)

        if (enter_pressed or send) and self.input_buffer.strip() and not self.busy:
            self.submit_prompt(self.input_buffer)
            self.input_buffer = ""

        imgui.end()

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

    def swap_clip(self, clip: MotionClip) -> None:
        seq_nodes = [n for n in self.scene.nodes if isinstance(n, SMPLSequence)]
        had_sequence = len(seq_nodes) > 0

        for node in seq_nodes:
            self.scene.remove(node)
        seq = smplx_params2_sequence(
            clip.smplx_params,
            num_betas=self.render_cfg.num_betas,
            betas=clip.betas,
            gender=self.render_cfg.gender,
            input_coord_system=clip.coord_system,
        )
        self.scene.add(seq)
        if not had_sequence:
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

        if self.stream_pipeline_runner is not None:
            threading.Thread(
                target=self.streaming_worker, args=(prompt,), daemon=True
            ).start()
        else:
            threading.Thread(target=self.full_clip_worker, args=(prompt,), daemon=True).start()

    def full_clip_worker(self, prompt: str) -> None:
        status = "ok"
        clip_result = None

        try:
            clip_result = self.pipeline_runner(prompt)

            if clip_result is None:
                status = "no motion"
        except Exception as exc:  # noqa: BLE001
            log.exception("[chat] pipeline failed")
            status = f"error: {exc}"

        with self.lock:
            self.pending_clip = clip_result

            if self.history and self.history[-1][0] == prompt:
                self.history[-1] = (prompt, status)

            self.busy = False
            self.busy_msg = ""

    def streaming_worker(self, prompt: str) -> None:
        assert self.stream_pipeline_runner is not None
        status = "ok"
        accumulated_frames: np.ndarray | None = None

        try:
            for chunk in self.stream_pipeline_runner(prompt):
                if accumulated_frames is None:
                    accumulated_frames = chunk
                else:
                    accumulated_frames = np.concatenate([accumulated_frames, chunk], axis=0)

                partial_clip = MotionClip(
                    action=prompt,
                    smplx_params=accumulated_frames.copy(),
                    coord_system=self.stream_coord_system,
                )
                with self.lock:
                    self.pending_clip = partial_clip
        except Exception as exc:  # noqa: BLE001
            log.exception("[chat] streaming pipeline failed")
            status = f"error: {exc}"

        with self.lock:
            if self.history and self.history[-1][0] == prompt:
                self.history[-1] = (prompt, status)

            self.busy = False
            self.busy_msg = ""

    def on_update(self) -> None:
        with self.lock:
            clip = self.pending_clip
            self.pending_clip = None
        if clip is not None:
            self.swap_clip(clip)

    def gui(self) -> None:
        self.on_update()
        super().gui()
