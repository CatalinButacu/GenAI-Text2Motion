from __future__ import annotations

import argparse
import time
import traceback
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Any

import imgui
import numpy as np
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer

from text2motion.studio.actions import log_action, log_result, log_toggle
from text2motion.studio.avatar import SmplxAvatarController, create_smplx_rest_pose_mesh
from text2motion.studio.config import StudioConfig
from text2motion.studio.contracts import MotionInferenceClientPort, ViewerHost
from text2motion.studio.generation import StudioGenerationController
from text2motion.studio.scene import StreamSceneUpdater, StudioSession

_COLOR_SLOTS = {
    "window_background": imgui.COLOR_WINDOW_BACKGROUND,
    "title_background": imgui.COLOR_TITLE_BACKGROUND,
    "title_background_active": imgui.COLOR_TITLE_BACKGROUND_ACTIVE,
    "frame_background": imgui.COLOR_FRAME_BACKGROUND,
    "frame_background_hovered": imgui.COLOR_FRAME_BACKGROUND_HOVERED,
    "frame_background_active": imgui.COLOR_FRAME_BACKGROUND_ACTIVE,
    "button": imgui.COLOR_BUTTON,
    "button_hovered": imgui.COLOR_BUTTON_HOVERED,
    "button_active": imgui.COLOR_BUTTON_ACTIVE,
    "header": imgui.COLOR_HEADER,
    "header_hovered": imgui.COLOR_HEADER_HOVERED,
    "header_active": imgui.COLOR_HEADER_ACTIVE,
    "slider_grab": imgui.COLOR_SLIDER_GRAB,
    "slider_grab_active": imgui.COLOR_SLIDER_GRAB_ACTIVE,
    "check_mark": imgui.COLOR_CHECK_MARK,
    "separator": imgui.COLOR_SEPARATOR,
    "plot_lines": imgui.COLOR_PLOT_LINES,
    "plot_histogram": imgui.COLOR_PLOT_HISTOGRAM,
}

_WINDOW_FLAGS = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE

_STYLE_FIELDS = (
    "window_rounding",
    "child_rounding",
    "frame_rounding",
    "grab_rounding",
    "popup_rounding",
    "scrollbar_rounding",
    "window_border_size",
    "window_padding",
    "frame_padding",
    "item_spacing",
)


class GuiPanels:
    def __init__(
        self,
        host: ViewerHost,
        state,
        avatar: SmplxAvatarController,
        generation: StudioGenerationController,
    ) -> None:
        self.host = host
        self.state = state
        self.avatar = avatar
        self.generation = generation

    def install(self) -> None:
        self.host.gui_controls["environment"] = self.gui_environment
        self.host.gui_controls["params"] = self.gui_params
        self.host.gui_controls["prompt"] = self.gui_prompt

    def _rects(self):
        layout = self.state.config.layout
        w, h = imgui.get_io().display_size
        top = imgui.get_frame_height()
        left_w = w * layout.left_width
        right_w = w * layout.right_width
        prompt_h = h * layout.prompt_height
        editor_h = (h - top) * layout.editor_height
        play_h = (h - top) * layout.playback_height
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
        theme = self.state.config.theme
        style = imgui.get_style()
        for name in _STYLE_FIELDS:
            setattr(style, name, getattr(theme, name))
        for name, rgba in theme.colors.items():
            slot = _COLOR_SLOTS.get(name)
            if slot is None:
                raise KeyError(
                    f"unknown imgui colour slot {name!r}; known slots: {sorted(_COLOR_SLOTS)}"
                )
            style.colors[slot] = rgba

    def _section(self, label: str) -> None:
        imgui.spacing()
        imgui.push_style_color(imgui.COLOR_TEXT, *self.state.config.theme.section_label_color)
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
        expanded, _ = imgui.begin("Scene", False, _WINDOW_FLAGS)
        if expanded:
            self.host.scene.gui_editor(imgui, self.host.viewports, self.host.viewport_mode)
        imgui.end()

    def gui_environment(self) -> None:
        _, env, *_ = self._rects()
        self._dock(env)
        imgui.begin("Environment", False, _WINDOW_FLAGS)

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
        changed, floor.enabled = imgui.checkbox("show floor", floor.enabled)
        if changed:
            log_toggle("show floor", floor.enabled)
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
        imgui.begin("Generation", False, _WINDOW_FLAGS)
        self._model_controls()
        self._sampling_controls()
        self._avatar_controls()
        self._prompt_history()
        imgui.end()

    def _model_controls(self) -> None:
        self._section("model")
        picked, self.state.model_idx = imgui.combo(
            "model", self.state.model_idx, [e.label for e in self.state.models]
        )
        if picked:
            log_action("select model", self.state.models[self.state.model_idx].key)
        entry = self.state.models[self.state.model_idx]
        self._dim_text(
            f"pinned from training: {entry.backbone} backbone, tokenizer"
            f" {Path(entry.tokenizer_ckpt).stem}, config {Path(entry.config).stem}"
        )
        busy = self.state.generating or self.state.loading_model
        if imgui.button("Load model", width=-1) and not busy:
            log_action("press Load model", self.state.models[self.state.model_idx].key)
            self.generation.start_model_load()
        if imgui.button("Reload settings + models (R)", width=-1) and not busy:
            log_action("press Reload settings")
            self.host.reload_settings()
        self._dim_text(self.state.load_status)

    def _sampling_controls(self) -> None:
        self._section("sampling")
        _, self.state.temperature = imgui.slider_float(
            "temperature", self.state.temperature, 0.1, 2.0
        )
        _, self.state.top_p = imgui.slider_float("top_p", self.state.top_p, 0.1, 1.0)
        _, self.state.cfg_scale = imgui.slider_float("cfg_scale", self.state.cfg_scale, 1.0, 12.0)
        if self.state.max_steps == 0:
            forever = self.state.steps == 0
            clicked, forever = imgui.checkbox("run forever (bounded state)", forever)
            if clicked:
                self.state.steps = 0 if forever else self.state.default_steps
                log_toggle("run forever", forever)
        if self.state.steps == 0:
            self._dim_text(
                "unbounded: this recurrent backbone streams until you press stop. "
                f"The viewer keeps the last {self.state.window_frames} frames."
            )
        else:
            upper = self.state.max_steps or self.state.window_steps
            _, self.state.steps = imgui.slider_int("steps (tokens)", self.state.steps, 8, upper)
            seconds = self.state.steps * self.state.downsample / self.host.playback_fps
            cap = (
                f"{self.state.max_steps} is the model's position-table hard cap."
                if self.state.max_steps
                else "this backbone has no position cap -- tick 'run forever' for unbounded."
            )
            trained_seconds = (
                self.state.trained_steps * self.state.downsample / self.host.playback_fps
            )
            self._dim_text(
                f"= {self.state.steps * self.state.downsample} frames (~{seconds:.1f}s @"
                f" {self.host.playback_fps:.0f} fps). Trained at {self.state.trained_steps} steps "
                f"(~{trained_seconds:.1f}s); much lower looks clipped; {cap}"
            )

    def _avatar_controls(self) -> None:
        self._section("avatar")
        clicked, self.state.fit_body = imgui.checkbox(
            "avatar skin (SMPL-X body)", self.state.fit_body
        )
        if clicked:
            log_toggle("avatar skin (fit body)", self.state.fit_body)
        if clicked and self.state.fit_body and not self.state.model_dir:
            self.state.fit_body = False
        elif clicked:
            self.avatar._on_fit_body_toggled()
        if not self.state.model_dir:
            self._dim_text("SMPL-X models not found: skeleton only.")
        for i, name in enumerate(self.state.config.genders):
            if imgui.radio_button(name, self.state.gender_idx == i) and self.state.gender_idx != i:
                self.state.gender_idx = i
                self.avatar._on_avatar_changed()
            if i < len(self.state.config.genders) - 1:
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
            for i in range(2, self.state.config.fit.num_betas):
                self._beta_slider(i, f"shape {i}")
        half = imgui.get_content_region_available()[0] / 2 - 4
        if imgui.button("reset shape", width=half) and self.state.user_betas.any():
            log_action("press reset shape")
            self.state.user_betas[:] = 0.0
            self.avatar._on_avatar_changed()
        imgui.same_line()
        if imgui.button("refresh avatar", width=half):
            log_action("press refresh avatar")
            self.avatar._on_avatar_changed()
        if self.state.smplx_fit.note:
            self._dim_text(self.state.smplx_fit.note)

    def _prompt_history(self) -> None:
        if len(self.state.history) > 0 and imgui.collapsing_header("prompt history")[0]:
            for i, past in enumerate(reversed(self.state.history[-10:])):
                clicked, _ = imgui.selectable(f"{past[:44]}##hist{i}")
                if clicked:
                    self.state.prompt_text = past

    def _beta_slider(self, i: int, label: str) -> None:
        changed, value = imgui.slider_float(label, float(self.state.user_betas[i]), -3.0, 3.0)
        if changed:
            self.state.user_betas[i] = value
        if imgui.is_item_deactivated_after_edit():
            self.avatar._on_avatar_changed()

    def gui_playback(self) -> None:
        *_, playback, _ = self._rects()
        self._dock(playback)
        imgui.begin("Playback", False, _WINDOW_FLAGS)

        changed, run = imgui.checkbox("play [space]", self.host.run_animations)
        if changed:
            log_toggle("play", run)
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

        changed, self.state.follow_cam = imgui.checkbox(
            "camera follows avatar", self.state.follow_cam
        )
        if changed:
            log_toggle("camera follows avatar", self.state.follow_cam)
        imgui.same_line()
        if imgui.small_button("center view"):
            self.state.stream_buffers.need_reset_view = True

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
        flags = _WINDOW_FLAGS | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR
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
                log_action("press Stop", f"at {self.state.stream_buffers.frames_emitted} frames")
                self.generation.stop()
        elif imgui.button("Generate", width=button_w) or enter:
            self.generation.submit_prompt()
        changed, self.state.live.enabled = imgui.checkbox(
            "live (regenerate as you type)", self.state.live.enabled
        )
        if changed:
            log_toggle("live regenerate", self.state.live.enabled)
        imgui.same_line()
        label = "hide skin (S)" if self.state.show_skin else "show skin (S)"
        if imgui.button(label):
            self.state.show_skin = not self.state.show_skin
            log_toggle("skin", self.state.show_skin)

        if self.state.generating:
            avail = imgui.get_content_region_available()[0]
            done = self.state.stream_buffers.frames_emitted
            total = self.state.steps * self.state.downsample
            label = f"motion {done}/{total} frames" if total else f"motion {done} frames (forever)"
            bar = self.state.stream_buffers.gen_progress if total else (done % 100) / 100.0
            if self.state.stream_buffers.current_fit_body:
                imgui.progress_bar(bar, (avail * 0.55, 15), label)
                imgui.same_line()
                fit = self.state.stream_buffers.fit_progress or 0.0
                imgui.progress_bar(fit, (-1, 15), f"body fit {int(fit * 100)}%")
            else:
                imgui.progress_bar(bar, (-1, 15), label)
        self._dim_text(self.state.status)
        imgui.end()


class StreamingStudioViewer(Viewer):
    scene: Any

    def __init__(
        self,
        client: MotionInferenceClientPort,
        model_dir: str,
        device: str,
        args: argparse.Namespace,
        config: StudioConfig,
        reload_config=None,
    ) -> None:
        super().__init__()
        self._reload_config = reload_config
        self.state = StudioSession(client, model_dir, device, args, self.scene, config)
        self.scene_updater = StreamSceneUpdater(self, self.state)
        self.avatar = SmplxAvatarController(self, self.state)
        self.generation = StudioGenerationController(self, self.state, self.avatar)
        self.panels = GuiPanels(self, self.state, self.avatar, self.generation)

        self.playback_fps = float(args.fps)
        self.run_animations = False
        self.scene.background_color = config.appearance.sky_color

        self.panels.install()
        self._add_rest_pose()

    def _add_rest_pose(self) -> None:
        state = self.state
        if not state.model_dir:
            return
        try:
            verts, faces = create_smplx_rest_pose_mesh(
                replace(state.config.fit, model_dir=state.model_dir, gender=state.gender), "cpu"
            )
            state.nodes.show_rest(
                Meshes(verts, faces, color=state.skin_color, name="Avatar (rest)")
            )
            state.stream_buffers.need_reset_view = True
        except Exception as exc:
            print(f"[warn] rest-pose SMPL-X unavailable ({exc}); skeleton-only until first prompt")
            state.model_dir = ""
            state.fit_body = False

    def gui_scene(self) -> None:
        self.panels.gui_scene()

    def gui_playback(self) -> None:
        self.panels.gui_playback()

    def reload_settings(self) -> None:
        if self._reload_config is None:
            self.state.load_status = "no config file to reload"
            return
        try:
            self.state.load_status = self.state.apply_config(self._reload_config())
            log_result("reload settings", True, self.state.load_status)
        except Exception as exc:
            traceback.print_exc()
            log_result("reload settings", False, f"{type(exc).__name__}: {exc}")
            self.state.load_status = f"reload failed: {type(exc).__name__}: {exc}"

    def key_event(self, key, action, modifiers) -> None:
        super().key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS or imgui.get_io().want_capture_keyboard:
            return
        if key == self.wnd.keys.S:
            self.state.show_skin = not self.state.show_skin
            log_toggle("skin [S]", self.state.show_skin)
        elif key == self.wnd.keys.R:
            log_action("key R")
            self.reload_settings()

    def on_render(self, time, frame_time, export=False, transparent_background=False) -> None:
        self.generation.tick_live()
        self.scene_updater._consume()
        self.scene_updater._sync_display_nodes()
        self.scene_updater._update_follow_cam()
        super().on_render(
            time, frame_time, export=export, transparent_background=transparent_background
        )
