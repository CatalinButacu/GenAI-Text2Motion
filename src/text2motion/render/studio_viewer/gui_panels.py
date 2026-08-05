from __future__ import annotations

import time
from array import array
from pathlib import Path

import imgui
import numpy as np

from text2motion.render.studio_viewer.constants import GENDERS, N_BETAS


class GuiPanels:
    def __init__(self, host, state, avatar, generation) -> None:
        self.host = host
        self.state = state
        self.avatar = avatar
        self.generation = generation

    def install(self) -> None:
        self.host.gui_controls["environment"] = self.gui_environment
        self.host.gui_controls["params"] = self.gui_params
        self.host.gui_controls["prompt"] = self.gui_prompt

    def _rects(self):
        w, h = imgui.get_io().display_size
        top = imgui.get_frame_height()
        left_w = w * 0.20
        right_w = w * 0.24
        prompt_h = h * 0.11
        editor_h = (h - top) * 0.52
        play_h = (h - top) * 0.30
        editor = (0.0, top, left_w, editor_h)
        env = (0.0, top + editor_h, left_w, h - top - editor_h)
        params = (w - right_w, top, right_w, h - top - play_h)
        playback = (w - right_w, h - play_h, right_w, play_h)
        prompt = (left_w, h - prompt_h, w - left_w - right_w, prompt_h)
        return editor, env, params, playback, prompt

    def _dock(self, rect) -> None:
        x, y, w, h = rect
        imgui.set_next_window_position(x, y, imgui.ALWAYS)
        imgui.set_next_window_size(w, h, imgui.ALWAYS)

    def _apply_style(self) -> None:
        if self.state.styled:
            return
        self.state.styled = True
        style = imgui.get_style()
        style.window_rounding = 8.0
        style.child_rounding = 6.0
        style.frame_rounding = 5.0
        style.grab_rounding = 5.0
        style.popup_rounding = 6.0
        style.scrollbar_rounding = 6.0
        style.window_border_size = 0.0
        style.window_padding = (12.0, 10.0)
        style.frame_padding = (8.0, 4.0)
        style.item_spacing = (8.0, 6.0)
        c = style.colors
        c[imgui.COLOR_WINDOW_BACKGROUND] = (0.09, 0.10, 0.13, 0.94)
        c[imgui.COLOR_TITLE_BACKGROUND] = (0.07, 0.08, 0.11, 1.0)
        c[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.12, 0.19, 0.32, 1.0)
        c[imgui.COLOR_FRAME_BACKGROUND] = (0.16, 0.18, 0.23, 1.0)
        c[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.22, 0.25, 0.32, 1.0)
        c[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.26, 0.30, 0.38, 1.0)
        c[imgui.COLOR_BUTTON] = (0.20, 0.34, 0.60, 1.0)
        c[imgui.COLOR_BUTTON_HOVERED] = (0.26, 0.44, 0.78, 1.0)
        c[imgui.COLOR_BUTTON_ACTIVE] = (0.30, 0.52, 0.92, 1.0)
        c[imgui.COLOR_HEADER] = (0.18, 0.26, 0.42, 1.0)
        c[imgui.COLOR_HEADER_HOVERED] = (0.24, 0.34, 0.54, 1.0)
        c[imgui.COLOR_HEADER_ACTIVE] = (0.28, 0.40, 0.62, 1.0)
        c[imgui.COLOR_SLIDER_GRAB] = (0.42, 0.62, 0.98, 1.0)
        c[imgui.COLOR_SLIDER_GRAB_ACTIVE] = (0.55, 0.72, 1.0, 1.0)
        c[imgui.COLOR_CHECK_MARK] = (0.42, 0.66, 1.0, 1.0)
        c[imgui.COLOR_SEPARATOR] = (0.25, 0.28, 0.36, 1.0)
        c[imgui.COLOR_PLOT_LINES] = (0.42, 0.66, 1.0, 1.0)
        c[imgui.COLOR_PLOT_HISTOGRAM] = (0.42, 0.66, 1.0, 1.0)

    def _section(self, label: str) -> None:
        imgui.spacing()
        imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.75, 1.0, 1.0)
        imgui.text(label.upper())
        imgui.pop_style_color()
        imgui.separator()

    def _dim_text(self, text: str) -> None:
        imgui.push_style_color(imgui.COLOR_TEXT, 0.62, 0.66, 0.74, 1.0)
        imgui.text_wrapped(text)
        imgui.pop_style_color()

    def gui_scene(self) -> None:
        self._apply_style()
        editor, *_ = self._rects()
        self._dock(editor)
        expanded, _ = imgui.begin("Scene", False, self.state.flags)
        if expanded:
            self.host.scene.gui_editor(imgui, self.host.viewports, self.host.viewport_mode)
        imgui.end()

    def gui_environment(self) -> None:
        _, env, *_ = self._rects()
        self._dock(env)
        imgui.begin("Environment", False, self.state.flags)

        self._section("scene")
        changed, color = imgui.color_edit4("background", *self.host.scene.background_color)
        if changed:
            self.host.scene.background_color = color
        _, self.host.scene.ambient_strength = imgui.slider_float(
            "ambient", self.host.scene.ambient_strength, 0.0, 4.0
        )
        imgui.text("mood:")
        imgui.same_line()
        if imgui.small_button("default"):
            self.host.scene.light_mode = "default"
        imgui.same_line()
        if imgui.small_button("dark"):
            self.host.scene.light_mode = "dark"
        imgui.same_line()
        if imgui.small_button("diffuse"):
            self.host.scene.light_mode = "diffuse"
        _, self.host.shadows_enabled = imgui.checkbox("shadows [S]", self.host.shadows_enabled)

        self._section("lights")
        for i, light in enumerate(self.host.scene.lights):
            flags = imgui.TREE_NODE_DEFAULT_OPEN if i == 0 else 0
            if imgui.tree_node(f"light {i + 1}##light{i}", flags):
                _, light.strength = imgui.slider_float(f"strength##l{i}", light.strength, 0.0, 10.0)
                changed, elevation = imgui.slider_float(
                    f"elevation##l{i}", light.elevation, -90.0, 90.0
                )
                if changed:
                    light.elevation = elevation
                changed, azimuth = imgui.slider_float(f"azimuth##l{i}", light.azimuth, 0.0, 360.0)
                if changed:
                    light.azimuth = azimuth
                changed, col = imgui.color_edit3(f"color##l{i}", *light.light_color[:3])
                if changed:
                    light.light_color = col
                _, light.shadow_enabled = imgui.checkbox(
                    f"cast shadows##l{i}", light.shadow_enabled
                )
                imgui.tree_pop()

        self._section("floor & origin")
        floor = self.host.scene.floor
        _, floor.enabled = imgui.checkbox("show floor", floor.enabled)
        changed, col = imgui.color_edit3("tile a", *[float(v) for v in floor.c1[:3]])
        if changed:
            floor.c1[:3] = col
        changed, col = imgui.color_edit3("tile b", *[float(v) for v in floor.c2[:3]])
        if changed:
            floor.c2[:3] = col
        _, self.host.scene.origin.enabled = imgui.checkbox(
            "origin axes", self.host.scene.origin.enabled
        )
        imgui.end()

    def gui_params(self) -> None:
        *_, params, _, _ = self._rects()
        self._dock(params)
        imgui.begin("Generation", False, self.state.flags)

        self._section("model")
        _, self.state.model_idx = imgui.combo(
            "model", self.state.model_idx, [e.label for e in self.state.models]
        )
        entry = self.state.models[self.state.model_idx]
        self._dim_text(
            f"pinned from training: {entry.backbone} backbone, tokenizer"
            f" {Path(entry.tokenizer_ckpt).stem}, config {Path(entry.config).stem}"
        )
        busy = self.state.generating or self.state.loading_model
        if imgui.button("Load model", width=-1) and not busy:
            self.generation._start_model_load()
        self._dim_text(self.state.load_status)

        self._section("sampling")
        _, self.state.temperature = imgui.slider_float(
            "temperature", self.state.temperature, 0.1, 2.0
        )
        _, self.state.top_p = imgui.slider_float("top_p", self.state.top_p, 0.1, 1.0)
        _, self.state.cfg_scale = imgui.slider_float("cfg_scale", self.state.cfg_scale, 1.0, 12.0)
        _, self.state.steps = imgui.slider_int(
            "steps (tokens)", self.state.steps, 8, self.state.max_steps
        )
        seconds = self.state.steps * self.state.downsample / self.host.playback_fps
        self._dim_text(
            f"= {self.state.steps * self.state.downsample} frames (~{seconds:.1f}s @"
            f" {self.host.playback_fps:.0f} fps). Trained at 49 steps (~9.8s); much lower looks"
            f" clipped; {self.state.max_steps} is the model's position-table hard cap."
        )

        self._section("avatar")
        clicked, self.state.fit_body = imgui.checkbox(
            "avatar skin (SMPL-X body)", self.state.fit_body
        )
        if clicked and self.state.fit_body and not self.state.model_dir:
            self.state.fit_body = False
        elif clicked:
            self.avatar._on_fit_body_toggled()
        if not self.state.model_dir:
            self._dim_text("SMPL-X models not found: skeleton only.")
        for i, name in enumerate(GENDERS):
            if imgui.radio_button(name, self.state.gender_idx == i) and self.state.gender_idx != i:
                self.state.gender_idx = i
                self.avatar._on_avatar_changed()
            if i < len(GENDERS) - 1:
                imgui.same_line()
        changed, col = imgui.color_edit3("skin", *self.state.skin_color[:3])
        if changed:
            self.state.skin_color = (col[0], col[1], col[2], 1.0)
            for node in (self.state.nodes.body, self.state.nodes.rest):
                if node is not None:
                    node.color = self.state.skin_color
        self._beta_slider(0, "height")
        self._beta_slider(1, "build")
        if imgui.collapsing_header("more shape dimensions")[0]:
            for i in range(2, N_BETAS):
                self._beta_slider(i, f"shape {i}")
        half = imgui.get_content_region_available()[0] / 2 - 4
        if imgui.button("reset shape", width=half) and self.state.user_betas.any():
            self.state.user_betas[:] = 0.0
            self.avatar._on_avatar_changed()
        imgui.same_line()
        if imgui.button("refresh avatar", width=half):
            self.avatar._on_avatar_changed()
        if self.state.fit.note:
            self._dim_text(self.state.fit.note)

        if len(self.state.history) > 0 and imgui.collapsing_header("prompt history")[0]:
            for i, past in enumerate(reversed(self.state.history[-10:])):
                clicked, _ = imgui.selectable(f"{past[:44]}##hist{i}")
                if clicked:
                    self.state.prompt_text = past
        imgui.end()

    def _beta_slider(self, i: int, label: str) -> None:
        changed, value = imgui.slider_float(label, float(self.state.user_betas[i]), -3.0, 3.0)
        if changed:
            self.state.user_betas[i] = value
        if imgui.is_item_deactivated_after_edit():
            self.avatar._on_avatar_changed()

    def gui_playback(self) -> None:
        *_, playback, _ = self._rects()
        self._dock(playback)
        imgui.begin("Playback", False, self.state.flags)

        changed, run = imgui.checkbox("play [space]", self.host.run_animations)
        if changed:
            self.host.toggle_animation(run)
        imgui.same_line()
        n_frames = max(1, self.host.scene.n_frames)
        self._dim_text(f"frame {self.host.scene.current_frame_id + 1}/{n_frames}")
        imgui.push_item_width(-1)
        moved, frame = imgui.slider_int(
            "##frame", self.host.scene.current_frame_id, 0, n_frames - 1
        )
        if moved:
            self.host.scene.current_frame_id = frame
        imgui.pop_item_width()
        _, self.host.playback_fps = imgui.drag_float(
            "speed (fps)",
            self.host.playback_fps,
            0.1,
            min_value=1.0,
            max_value=120.0,
            format="%.1f",
        )

        _, self.state.follow_cam = imgui.checkbox("camera follows avatar", self.state.follow_cam)
        imgui.same_line()
        if imgui.small_button("center view"):
            self.state.buf.need_reset_view = True

        valid = self.host._past_frametimes[self.host._past_frametimes > 0.0]
        if valid.size > 0:
            fps_avg = 1.0 / float(np.mean(valid))
            imgui.plot_lines(
                f"render {fps_avg:.0f} fps",
                array("f", (1.0 / valid).tolist()),
                scale_min=0.0,
                scale_max=120.0,
                graph_size=(imgui.get_content_region_available()[0] - 110, 24),
            )
        imgui.end()

    def gui_prompt(self) -> None:
        *_, prompt = self._rects()
        self._dock(prompt)
        flags = self.state.flags | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR
        imgui.begin("##prompt_bar", False, flags)

        button_w = 96.0
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
        imgui.same_line()
        if self.state.generating and not self.state.live.enabled:
            if imgui.button("Stop", width=button_w):
                self.state.buf.cancel = True
        elif imgui.button("Generate", width=button_w) or enter:
            self.generation.submit_prompt()
        _, self.state.live.enabled = imgui.checkbox(
            "live (regenerate as you type)", self.state.live.enabled
        )

        if self.state.generating:
            avail = imgui.get_content_region_available()[0]
            done = int(self.state.buf.gen_progress * self.state.steps * self.state.downsample)
            total = self.state.steps * self.state.downsample
            if self.state.buf.current_fit_body:
                imgui.progress_bar(
                    self.state.buf.gen_progress, (avail * 0.55, 15), f"motion {done}/{total} frames"
                )
                imgui.same_line()
                fit = self.state.buf.fit_progress or 0.0
                imgui.progress_bar(fit, (-1, 15), f"body fit {int(fit * 100)}%")
            else:
                imgui.progress_bar(
                    self.state.buf.gen_progress, (-1, 15), f"motion {done}/{total} frames"
                )
        self._dim_text(self.state.status)
        imgui.end()
