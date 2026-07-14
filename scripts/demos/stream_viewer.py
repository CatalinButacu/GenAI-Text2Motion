from __future__ import annotations

import argparse
import threading
import time
import traceback
from array import array
from dataclasses import dataclass
from pathlib import Path

import imgui
import numpy as np
import torch
import yaml
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import ViewerCamera
from aitviewer.viewer import Viewer

from text2motion.render.joints2smpl import (
    FitConfig,
    build_smplx_model,
    fit_smplx_to_joints,
    mesh_from_params,
    rest_pose_body,
)
from text2motion.render.studio import build_skeleton_seq
from text2motion.shared.seed import seed_everything
from text2motion.stream.service import DEFAULT_HOST, DEFAULT_PORT, connect_or_spawn

SKIN_COLOR = (0.86, 0.72, 0.61, 1.0)  # SMPL-X body tone
SKY_COLOR = (240 / 255, 182 / 255, 182 / 255, 1.0)  # default background: soft pink
GENDERS = ("neutral", "male", "female")
BACKBONES = ("transformer", "mamba")
N_BETAS = 10  # SMPL-X shape dims exposed as avatar-dimension sliders
LIVE_STEPS = 20  # short clip while typing: fewer tokens -> faster preview than the full 49
LIVE_DEBOUNCE_S = 0.35  # wait this long after the last keystroke before regenerating
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = str(ROOT / "data" / "smplx_models")


@dataclass(frozen=True)
class ModelEntry:
    label: str
    config: str
    ckpt: str
    tokenizer_ckpt: str
    backbone: str


MODEL_REGISTRY = ROOT / "configs" / "demo_models.yaml"


def load_model_registry(launch: ModelEntry) -> tuple[list[ModelEntry], int]:
    entries: list[ModelEntry] = []
    if MODEL_REGISTRY.is_file():
        raw = yaml.safe_load(MODEL_REGISTRY.read_text(encoding="utf-8"))
        for name, e in raw.items():
            entries.append(
                ModelEntry(
                    label=e.get("label", name),
                    config=e["config"],
                    ckpt=e["ckpt"],
                    tokenizer_ckpt=e["tokenizer_ckpt"],
                    backbone=e["backbone"],
                )
            )
    else:
        print(f"[warn] model registry not found: {MODEL_REGISTRY} -> launch args only")
    launch_ckpt = Path(launch.ckpt).resolve()
    for i, e in enumerate(entries):
        if Path(e.ckpt).resolve() == launch_ckpt:
            return entries, i
    entries.insert(0, launch)
    return entries, 0


class StreamingStudioViewer(Viewer):
    def __init__(self, client, model_dir: str, device: str, args: argparse.Namespace) -> None:
        super().__init__()
        self.client = (
            client  # generation runs in the motion-service PROCESS (own GIL -> smooth GUI)
        )
        self.downsample = int(client.hello["downsample"])
        self.device = device  # local GPU: SMPL-X fitting + rendering only
        self.model_dir = model_dir  # "" -> SMPL-X unavailable: skeleton-only mode

        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.cfg_scale = float(args.cfg_scale)
        self.steps = int(args.steps)

        self.gender_idx = 0
        self.fit_body = bool(model_dir)
        self.skin_color = SKIN_COLOR
        self.user_betas = np.zeros(N_BETAS, dtype=np.float32)  # offsets on the fitted shape

        self.backbone = args.backbone
        launch_entry = ModelEntry(
            label=f"launch args: {Path(args.ckpt).stem}",
            config=args.config,
            ckpt=args.ckpt,
            tokenizer_ckpt=args.tokenizer_ckpt,
            backbone=args.backbone,
        )
        self._models, self._model_idx = load_model_registry(launch_entry)
        self._loading_model = False
        self.load_status = f"active: {self._models[self._model_idx].label}"

        self.prompt_text = ""
        self.status = "ready -- describe a motion below and press Enter"
        self.generating = False
        self.history: list[str] = []

        self.live_mode = False  # regenerate a fast preview on every (debounced) text change
        self._prev_prompt = ""
        self._prompt_dirty_at = 0.0
        self._regen_pending = False  # a newer prompt arrived mid-generation: restart once it stops
        self._last_live_prompt = ""
        self._sampling_sig = (
            self.temperature,
            self.top_p,
            self.steps,
        )  # live-mode watches these too

        self.lock = threading.Lock()
        self._new_chunks: list[np.ndarray] = []
        self._pending_body_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self._pending_remesh: tuple[np.ndarray, np.ndarray, bool] | None = None
        self._pending_pipeline: tuple | None = None
        self._current_fit_body = False  # fit_body flag for the in-flight/most-recent generation
        self._drop_skeleton = False  # generation over: retire the live skeleton, body remains
        self._smplx_model_cache: dict[tuple[int, str], object] = {}  # (T, gender) -> built model
        self._accum_joints: np.ndarray | None = None
        self._body_verts: np.ndarray | None = None
        self._started = False
        self._need_reset_view = False
        self._cancel = False

        self._fit_orient: list[np.ndarray] = []
        self._fit_pose: list[np.ndarray] = []
        self._fit_transl: list[np.ndarray] = []
        self._fit_betas: np.ndarray | None = None
        self._remesh_version = 0
        self._remesh_done_version = 0
        self._remesh_running = False
        self._remesh_after_gen = False
        self._remesh_note = ""

        self._gen_progress = 0.0
        self._fit_progress: float | None = None

        self._rest_node = None
        self._motion_node = None
        self._body_node = None

        self.follow_cam = True
        self._follow_center: np.ndarray | None = None

        self._flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE
        self._styled = False
        self.playback_fps = float(args.fps)
        self.run_animations = False
        self.scene.background_color = SKY_COLOR

        self._install_panels()
        self._add_rest_pose()

    def _install_panels(self) -> None:
        self.gui_controls["environment"] = self.gui_environment
        self.gui_controls["params"] = self.gui_params
        self.gui_controls["prompt"] = self.gui_prompt

    def _add_rest_pose(self) -> None:
        if not self.model_dir:
            return
        try:
            verts, faces = rest_pose_body(
                FitConfig(model_dir=self.model_dir, gender=self.gender), "cpu"
            )
            self._rest_node = Meshes(verts, faces, color=self.skin_color, name="Avatar (rest)")
            self.scene.add(self._rest_node)
            self._need_reset_view = True
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the studio
            print(f"[warn] rest-pose SMPL-X unavailable ({exc}); skeleton-only until first prompt")
            self.model_dir = ""
            self.fit_body = False

    @property
    def gender(self) -> str:
        return GENDERS[self.gender_idx]

    def _rects(self):
        w, h = imgui.get_io().display_size
        top = imgui.get_frame_height()  # main menu bar height, tracks the scaled font
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
        if self._styled:
            return
        self._styled = True
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
        expanded, _ = imgui.begin("Scene", False, self._flags)
        if expanded:
            self.scene.gui_editor(imgui, self.viewports, self.viewport_mode)
        imgui.end()

    def gui_environment(self) -> None:
        _, env, *_ = self._rects()
        self._dock(env)
        imgui.begin("Environment", False, self._flags)

        self._section("scene")
        changed, color = imgui.color_edit4("background", *self.scene.background_color)
        if changed:
            self.scene.background_color = color
        _, self.scene.ambient_strength = imgui.slider_float(
            "ambient", self.scene.ambient_strength, 0.0, 4.0
        )
        imgui.text("mood:")
        imgui.same_line()
        if imgui.small_button("default"):
            self.scene.light_mode = "default"
        imgui.same_line()
        if imgui.small_button("dark"):
            self.scene.light_mode = "dark"
        imgui.same_line()
        if imgui.small_button("diffuse"):
            self.scene.light_mode = "diffuse"
        _, self.shadows_enabled = imgui.checkbox("shadows [S]", self.shadows_enabled)

        self._section("lights")
        for i, light in enumerate(self.scene.lights):
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
        floor = self.scene.floor
        _, floor.enabled = imgui.checkbox("show floor", floor.enabled)
        changed, col = imgui.color_edit3("tile a", *[float(v) for v in floor.c1[:3]])
        if changed:
            floor.c1[:3] = col
        changed, col = imgui.color_edit3("tile b", *[float(v) for v in floor.c2[:3]])
        if changed:
            floor.c2[:3] = col
        _, self.scene.origin.enabled = imgui.checkbox("origin axes", self.scene.origin.enabled)
        imgui.end()

    def gui_params(self) -> None:
        *_, params, _, _ = self._rects()
        self._dock(params)
        imgui.begin("Generation", False, self._flags)

        self._section("model")
        _, self._model_idx = imgui.combo("model", self._model_idx, [e.label for e in self._models])
        entry = self._models[self._model_idx]
        self._dim_text(
            f"pinned from training: {entry.backbone} backbone, tokenizer"
            f" {Path(entry.tokenizer_ckpt).stem}, config {Path(entry.config).stem}"
        )
        busy = self.generating or self._loading_model
        if imgui.button("Load model", width=-1) and not busy:
            self._start_model_load()
        self._dim_text(self.load_status)

        self._section("sampling")
        _, self.temperature = imgui.slider_float("temperature", self.temperature, 0.1, 2.0)
        _, self.top_p = imgui.slider_float("top_p", self.top_p, 0.1, 1.0)
        _, self.cfg_scale = imgui.slider_float("cfg_scale", self.cfg_scale, 1.0, 12.0)
        _, self.steps = imgui.slider_int("steps (tokens)", self.steps, 8, 100)
        seconds = self.steps * self.downsample / self.playback_fps
        self._dim_text(
            f"= {self.steps * self.downsample} frames (~{seconds:.1f}s @"
            f" {self.playback_fps:.0f} fps). Trained at 49 steps (~9.8s); much lower looks"
            " clipped, much higher is out-of-distribution."
        )

        self._section("avatar")
        clicked, self.fit_body = imgui.checkbox("avatar skin (SMPL-X body)", self.fit_body)
        if clicked and self.fit_body and not self.model_dir:
            self.fit_body = False
        elif clicked:
            self._on_fit_body_toggled()
        if not self.model_dir:
            self._dim_text("SMPL-X models not found: skeleton only.")
        for i, name in enumerate(GENDERS):
            if imgui.radio_button(name, self.gender_idx == i) and self.gender_idx != i:
                self.gender_idx = i
                self._on_avatar_changed()
            if i < len(GENDERS) - 1:
                imgui.same_line()
        changed, col = imgui.color_edit3("skin", *self.skin_color[:3])
        if changed:
            self.skin_color = (col[0], col[1], col[2], 1.0)
            for node in (self._body_node, self._rest_node):
                if node is not None:
                    node.color = self.skin_color
        self._beta_slider(0, "height")
        self._beta_slider(1, "build")
        if imgui.collapsing_header("more shape dimensions")[0]:
            for i in range(2, N_BETAS):
                self._beta_slider(i, f"shape {i}")
        half = imgui.get_content_region_available()[0] / 2 - 4
        if imgui.button("reset shape", width=half) and self.user_betas.any():
            self.user_betas[:] = 0.0
            self._on_avatar_changed()
        imgui.same_line()
        if imgui.button("refresh avatar", width=half):
            self._on_avatar_changed()
        if self._remesh_note:
            self._dim_text(self._remesh_note)

        if len(self.history) > 0 and imgui.collapsing_header("prompt history")[0]:
            for i, past in enumerate(reversed(self.history[-10:])):
                clicked, _ = imgui.selectable(f"{past[:44]}##hist{i}")
                if clicked:
                    self.prompt_text = past
        imgui.end()

    def _beta_slider(self, i: int, label: str) -> None:
        changed, value = imgui.slider_float(label, float(self.user_betas[i]), -3.0, 3.0)
        if changed:
            self.user_betas[i] = value
        if imgui.is_item_deactivated_after_edit():
            self._on_avatar_changed()

    def gui_playback(self) -> None:
        *_, playback, _ = self._rects()
        self._dock(playback)
        imgui.begin("Playback", False, self._flags)

        changed, run = imgui.checkbox("play [space]", self.run_animations)
        if changed:
            self.toggle_animation(run)
        imgui.same_line()
        n_frames = max(1, self.scene.n_frames)
        self._dim_text(f"frame {self.scene.current_frame_id + 1}/{n_frames}")
        imgui.push_item_width(-1)
        moved, frame = imgui.slider_int("##frame", self.scene.current_frame_id, 0, n_frames - 1)
        if moved:
            self.scene.current_frame_id = frame
        imgui.pop_item_width()
        _, self.playback_fps = imgui.drag_float(
            "speed (fps)", self.playback_fps, 0.1, min_value=1.0, max_value=120.0, format="%.1f"
        )

        _, self.follow_cam = imgui.checkbox("camera follows avatar", self.follow_cam)
        imgui.same_line()
        if imgui.small_button("center view"):
            self._need_reset_view = True

        valid = self._past_frametimes[self._past_frametimes > 0.0]
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
        flags = self._flags | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR
        imgui.begin("##prompt_bar", False, flags)

        button_w = 96.0
        imgui.push_item_width(imgui.get_content_region_available()[0] - button_w - 8)
        enter, self.prompt_text = imgui.input_text_with_hint(
            "##prompt",
            'describe a motion, e.g. "a person walks in a circle" -- Enter to generate',
            self.prompt_text,
            256,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        if self.prompt_text != self._prev_prompt:  # a keystroke: (re)start the debounce clock
            self._prev_prompt = self.prompt_text
            self._prompt_dirty_at = time.perf_counter()
        imgui.pop_item_width()
        imgui.same_line()
        if self.generating and not self.live_mode:
            if imgui.button("Stop", width=button_w):
                self._cancel = True
        elif imgui.button("Generate", width=button_w) or enter:
            self.submit_prompt()
        _, self.live_mode = imgui.checkbox("live (regenerate as you type)", self.live_mode)

        if self.generating:
            avail = imgui.get_content_region_available()[0]
            done = int(self._gen_progress * self.steps * self.downsample)
            total = self.steps * self.downsample
            if self._current_fit_body:
                imgui.progress_bar(
                    self._gen_progress, (avail * 0.55, 15), f"motion {done}/{total} frames"
                )
                imgui.same_line()
                fit = self._fit_progress or 0.0
                imgui.progress_bar(fit, (-1, 15), f"body fit {int(fit * 100)}%")
            else:
                imgui.progress_bar(self._gen_progress, (-1, 15), f"motion {done}/{total} frames")
        self._dim_text(self.status)
        imgui.end()

    def submit_prompt(self) -> None:
        prompt = self.prompt_text.strip()
        if not prompt or self.generating:
            return
        self._last_live_prompt = prompt  # a manual Generate also satisfies the live watcher
        self._start_generation(prompt, live=False)

    def _start_generation(self, prompt: str, live: bool) -> None:
        self.generating = True
        self._cancel = False
        self._gen_progress = 0.0
        self._fit_progress = None
        if live:
            steps = min(self.steps, LIVE_STEPS)  # short preview clip
            cfg_scale = 1.0  # cfg 1.0 -> single forward/step (~2x faster; no unconditional branch)
            fit_body = False  # skeleton only: skip the seconds-per-chunk SMPL-X fit
            self.status = f'live: "{prompt}"'
        else:
            steps = self.steps
            cfg_scale = self.cfg_scale
            fit_body = self.fit_body and bool(self.model_dir)
            self.status = f'generating: "{prompt}"'
            self.history.append(prompt)
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "fit_body": fit_body,
            "downsample": self.downsample,
        }
        self._current_fit_body = fit_body
        self._drop_skeleton = False
        with self.lock:
            self._new_chunks.clear()
            self._pending_body_chunks = []
            self._pending_remesh = None
            self._started = False
            self._accum_joints = None
            self._fit_orient = []
            self._fit_pose = []
            self._fit_transl = []
            self._fit_betas = None
        threading.Thread(target=self._worker, args=(prompt, params), daemon=True).start()

    def _tick_live(self) -> None:
        if not self.live_mode:
            return
        sig = (self.temperature, self.top_p, self.steps)  # cfg is forced to 1.0 in live preview
        if sig != self._sampling_sig:  # a sampling slider moved: re-run current prompt like an edit
            self._sampling_sig = sig
            if self.prompt_text.strip():
                self._prev_prompt = self.prompt_text
                self._prompt_dirty_at = time.perf_counter()
                self._last_live_prompt = ""
        prompt = self.prompt_text.strip()
        ready = (
            self._prev_prompt != ""
            and (time.perf_counter() - self._prompt_dirty_at) >= LIVE_DEBOUNCE_S
        )
        if prompt and ready and prompt != self._last_live_prompt:
            self._prev_prompt = ""  # consume this edit so we fire once per settle
            self._last_live_prompt = prompt
            if self.generating:
                self._cancel = True  # abandon the stale preview, restart when its worker stops
                self._regen_pending = True
            else:
                self._start_generation(prompt, live=True)
        if self._regen_pending and not self.generating:
            self._regen_pending = False
            if self.prompt_text.strip():
                self._start_generation(self.prompt_text.strip(), live=True)

    def _worker(self, prompt: str, params: dict) -> None:
        n_frames = 0
        total_frames = params["steps"] * params["downsample"]
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        try:
            print(
                f"[GEN] request prompt={prompt!r} steps={params['steps']} "
                f"cfg={params['cfg_scale']} temp={params['temperature']} "
                f"fit_body={params['fit_body']} -> motion service",
                flush=True,
            )
            chunk_iter = self.client.generate(
                prompt,
                steps=params["steps"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                cfg_scale=params["cfg_scale"],
                should_cancel=lambda: self._cancel,
            )
            for joints in chunk_iter:  # (t, 22, 3) float32, produced in the service process
                assert joints.ndim == 3 and joints.shape[1:] == (22, 3), (
                    f"bad joints {joints.shape}"
                )
                n_frames += len(joints)
                print(
                    f"[GEN] chunk joints={joints.shape} n_frames={n_frames} "
                    f"finite={np.isfinite(joints).all()}",
                    flush=True,
                )
                self._gen_progress = min(1.0, n_frames / total_frames)
                with self.lock:
                    self._new_chunks.append(joints)
                if params["fit_body"]:
                    self.status = f"streaming... {n_frames} frames"
                    warm_start = self._fit_chunk(joints, warm_start, n_frames)
                else:
                    self.status = (
                        f"streaming... {n_frames} frames (skeleton only -- tick"
                        " 'avatar skin' for the body)"
                    )
        except Exception as exc:  # noqa: BLE001 - full traceback to console; short summary in the GUI
            traceback.print_exc()
            self.status = f"error after {n_frames} frames: {type(exc).__name__}: {exc}"
            self.generating = False
            return

        print(
            f"[GEN] worker DONE total_frames={n_frames} fit_body={params['fit_body']}", flush=True
        )
        duration = n_frames / self.playback_fps
        verb = "stopped" if self._cancel else "done"
        suffix = "" if params["fit_body"] else ", skeleton"
        self.status = f"{verb} -- {n_frames} frames (~{duration:.1f}s{suffix})"
        self._fit_progress = None
        if params["fit_body"] and not self._cancel:
            self._drop_skeleton = True  # the fitted body is complete: retire the live skeleton
        if not self.run_animations:
            self.toggle_animation(True)  # finished motions always loop
        self.generating = False
        if params["fit_body"] and self._remesh_after_gen and self._fit_orient:
            self._remesh_after_gen = False
            self._schedule_remesh()

    def _fit_chunk(
        self,
        joints: np.ndarray,
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        n_frames_so_far: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        is_first = warm_start is None
        gender = self.gender
        cfg = FitConfig(
            model_dir=self.model_dir,
            gender=gender,
            stage1_iters=60 if is_first else 0,
            stage2_iters=150 if is_first else 25,
        )
        duration_so_far = n_frames_so_far / self.playback_fps
        t = joints.shape[0]
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()

            model = self._smplx_model_cache.get((t, gender))
            if model is None:
                model = build_smplx_model(cfg, t, self.device)
                self._smplx_model_cache[(t, gender)] = model

            with torch.enable_grad():
                res = fit_smplx_to_joints(
                    joints,
                    cfg,
                    device=self.device,
                    warm_start=warm_start,
                    model=model,
                    on_progress=self._on_fit_progress,
                )

            display_verts = self._shaped_vertices(model, res, t)
            with self.lock:
                self._pending_body_chunks.append((display_verts, res.faces))
                self._fit_orient.append(res.global_orient)
                self._fit_pose.append(res.body_pose)
                self._fit_transl.append(res.transl)
                if self._fit_betas is None:
                    self._fit_betas = res.betas

            self.status = (
                f"streaming... {n_frames_so_far} frames (~{duration_so_far:.1f}s), "
                f"body fit {res.joint_err_cm:.1f} cm"
            )
            return (res.global_orient[-1], res.body_pose[-1], res.betas)
        except Exception as exc:  # noqa: BLE001 - skip this chunk's body fit, keep streaming
            traceback.print_exc()
            self.status = (
                f"streaming... {n_frames_so_far} frames but SMPL-X fit failed for a chunk: "
                f"{type(exc).__name__}: {exc}"
            )
            return warm_start

    def _shaped_vertices(self, model, res, t: int) -> np.ndarray:
        user = self.user_betas
        if not user.any():
            return res.vertices
        dev = self.device if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            out = model(
                global_orient=torch.tensor(res.global_orient, device=dev),
                body_pose=torch.tensor(res.body_pose, device=dev),
                transl=torch.tensor(res.transl, device=dev),
                betas=torch.tensor(res.betas + user, dtype=torch.float32, device=dev)
                .unsqueeze(0)
                .expand(t, -1),
            )
        return out.vertices.cpu().numpy()

    def _on_fit_progress(self, done: int, total: int) -> None:
        self._fit_progress = done / max(1, total)

    def _on_avatar_changed(self) -> None:
        if not self.model_dir or not self.fit_body:
            return
        if self.generating:
            self._remesh_after_gen = True
            self._remesh_note = "new chunks use the new avatar; full clip refreshes when done"
            return
        self._schedule_remesh()

    def _on_fit_body_toggled(self) -> None:
        if self.generating:
            self._remesh_note = "body-fit choice applies to the next generation"
            return
        if not self.fit_body:
            self._remove("_body_node")
            self._remove("_rest_node")
            if self._accum_joints is not None and self._motion_node is None:
                self._motion_node = build_skeleton_seq(self._accum_joints)
                self._motion_node.name = "Motion (streaming)"
                self.scene.add(self._motion_node)
            return
        if self._accum_joints is None or self._fit_orient:
            self._schedule_remesh()  # rest pose, or re-dress from the already-fitted params
        else:
            self._start_retrofit()  # motion exists but was generated skeleton-only: fit it now

    def _start_retrofit(self) -> None:
        joints = self._accum_joints
        if joints is None or self.generating:
            return
        self.generating = True
        self._cancel = False
        self._current_fit_body = True
        self._gen_progress = 1.0  # the motion itself is already fully decoded
        self._drop_skeleton = False
        self.status = "fitting SMPL-X to the generated motion..."
        with self.lock:
            self._pending_body_chunks = []
            self._fit_orient = []
            self._fit_pose = []
            self._fit_transl = []
            self._fit_betas = None
        threading.Thread(target=self._retrofit_worker, args=(joints.copy(),), daemon=True).start()

    def _retrofit_worker(self, joints: np.ndarray) -> None:
        warm_start = None
        window = 16
        n_frames = 0
        try:
            for start in range(0, len(joints), window):
                if self._cancel:
                    break
                piece = joints[start : start + window]
                n_frames += len(piece)
                warm_start = self._fit_chunk(piece, warm_start, n_frames)
        finally:
            verb = "stopped" if self._cancel else "done"
            self.status = f"body fit {verb} -- {n_frames} frames"
            self._fit_progress = None
            if not self._cancel:
                self._drop_skeleton = True
            if not self.run_animations:
                self.toggle_animation(True)  # finished motions always loop
            self.generating = False

    def _schedule_remesh(self) -> None:
        self._remesh_version += 1
        if self._remesh_running:
            return
        self._remesh_running = True
        threading.Thread(target=self._remesh_worker, daemon=True).start()

    def _remesh_worker(self) -> None:
        try:
            while self._remesh_done_version != self._remesh_version:
                version = self._remesh_version
                gender = self.gender
                user = self.user_betas.copy()
                with self.lock:
                    orient = np.concatenate(self._fit_orient, 0) if self._fit_orient else None
                    pose = np.concatenate(self._fit_pose, 0) if self._fit_pose else None
                    transl = np.concatenate(self._fit_transl, 0) if self._fit_transl else None
                    fit_betas = None if self._fit_betas is None else self._fit_betas.copy()
                cfg = FitConfig(model_dir=self.model_dir, gender=gender)
                try:
                    self._remesh_note = f"updating avatar ({gender})..."
                    if orient is None:
                        zero = np.zeros((1, 3), dtype=np.float32)
                        verts, faces = mesh_from_params(
                            cfg,
                            zero,
                            np.zeros((1, 63), dtype=np.float32),
                            zero,
                            user,
                            gender,
                            "cpu",
                        )
                        is_rest = True
                    else:
                        betas = user if fit_betas is None else fit_betas + user
                        verts, faces = mesh_from_params(
                            cfg,
                            orient,
                            pose,
                            transl,
                            betas.astype(np.float32),
                            gender,
                            self.device,
                        )
                        is_rest = False
                    with self.lock:
                        self._pending_remesh = (verts, faces, is_rest)
                    self._remesh_note = ""
                except Exception as exc:  # noqa: BLE001 - GUI note + console traceback
                    traceback.print_exc()
                    self._remesh_note = f"avatar update failed: {type(exc).__name__}: {exc}"
                self._remesh_done_version = version
        finally:
            self._remesh_running = False
        if self._remesh_done_version != self._remesh_version:  # request landed while finishing
            self._schedule_remesh()

    def _start_model_load(self) -> None:
        entry = self._models[self._model_idx]
        self._loading_model = True
        self.load_status = f"loading {entry.label}..."
        threading.Thread(target=self._load_worker, args=(entry,), daemon=True).start()

    def _load_worker(self, entry: ModelEntry) -> None:
        try:
            hello = self.client.load(
                entry.config, entry.ckpt, entry.tokenizer_ckpt, entry.backbone
            )  # the service swaps the model in ITS process; the viewer stays light
            self.backbone = entry.backbone
            self.downsample = int(hello["downsample"])
            self.load_status = f"active: {entry.label}"
        except Exception as exc:  # noqa: BLE001 - a missing/mismatched bundle must not crash the studio
            traceback.print_exc()
            self.load_status = f"load failed: {type(exc).__name__}: {exc}"
        finally:
            self._loading_model = False

    def on_render(self, time, frame_time, **kwargs) -> None:
        self._tick_live()
        self._consume()
        self._sync_display_nodes()
        self._update_follow_cam()
        super().on_render(time, frame_time, **kwargs)

    def _sync_display_nodes(self) -> None:
        if self._motion_node is None or self._body_node is None:
            return
        body_frames = 0 if self._body_verts is None else len(self._body_verts)
        on_body = self.scene.current_frame_id < body_frames
        self._body_node.enabled = on_body
        self._motion_node.enabled = not on_body

    def _consume(self) -> None:
        with self.lock:
            chunks = self._new_chunks
            self._new_chunks = []
            body_chunks = self._pending_body_chunks
            self._pending_body_chunks = []
            remesh = self._pending_remesh
            self._pending_remesh = None

        if chunks:
            self._append_chunks(chunks)
        if body_chunks:
            self._append_body_chunks(body_chunks)
        if self._drop_skeleton and not body_chunks:  # all fitted chunks landed: body takes over
            self._drop_skeleton = False
            if self._body_node is not None:
                self._remove("_motion_node")
                self._body_node.enabled = True
        if remesh is not None:
            self._apply_remesh(*remesh)
        if self._need_reset_view:
            self._need_reset_view = False
            node = self._body_node or self._motion_node or self._rest_node
            if node is not None:
                try:
                    self.center_view_on_node(node)
                except Exception:  # noqa: BLE001 - camera framing is best-effort
                    pass

    def _remove(self, attr: str) -> None:
        node = getattr(self, attr)
        if node is not None:
            if (
                self.scene.selected_object is node
            ):  # else the outline pass renders a removed VAO -> crash
                self.scene.select(None)
            try:
                self.scene.remove(node)
            except Exception:  # noqa: BLE001
                pass
            setattr(self, attr, None)

    def _append_chunks(self, chunks: list[np.ndarray]) -> None:
        new = np.concatenate(chunks, axis=0)
        self._accum_joints = (
            new if self._accum_joints is None else np.concatenate([self._accum_joints, new], axis=0)
        )
        self._remove("_motion_node")
        self._motion_node = build_skeleton_seq(self._accum_joints)
        self._motion_node.name = "Motion (streaming)"
        self.scene.add(self._motion_node)
        print(
            f"[DISP] skeleton node added: accum={self._accum_joints.shape} "
            f"scene_frames={self.scene.n_frames} started={self._started}",
            flush=True,
        )
        if not self._started:
            self._started = True
            self._remove("_body_node")
            self._remove("_rest_node")
            self._follow_center = None  # re-anchor the follow-cam on the new clip
            self.scene.current_frame_id = 0
            self.toggle_animation(True)
            self._need_reset_view = True

    def _append_body_chunks(self, chunks: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for verts, faces in chunks:
            if self._body_node is None:
                self._body_node = Meshes(
                    verts, faces, color=self.skin_color, name="Avatar (SMPL-X)"
                )
                self.scene.add(self._body_node)
                self._body_verts = verts
            else:
                self._body_node.add_frames(verts)
                self._body_verts = self._body_node.vertices

    def _apply_remesh(self, verts: np.ndarray, faces: np.ndarray, is_rest: bool) -> None:
        if is_rest:
            if self._body_node is not None or self._motion_node is not None:
                return  # a motion arrived while re-meshing the rest pose; keep the motion
            self._remove("_rest_node")
            self._rest_node = Meshes(verts, faces, color=self.skin_color, name="Avatar (rest)")
            self.scene.add(self._rest_node)
            return
        frame = self.scene.current_frame_id
        self._remove("_motion_node")
        self._remove("_rest_node")
        self._remove("_body_node")
        self._body_node = Meshes(verts, faces, color=self.skin_color, name="Avatar (SMPL-X)")
        self.scene.add(self._body_node)
        self._body_verts = verts
        self.scene.current_frame_id = min(frame, len(verts) - 1)

    def _current_center(self) -> np.ndarray | None:
        frame = self.scene.current_frame_id
        body_frames = 0 if self._body_verts is None else len(self._body_verts)
        if self._body_node is not None and (self._motion_node is None or frame < body_frames):
            arr = self._body_verts  # the body is the visible figure at this frame
        elif self._motion_node is not None and self._accum_joints is not None:
            arr = self._accum_joints  # not-yet-fitted tail: the skeleton is the visible figure
        else:
            return None
        return arr[min(frame, len(arr) - 1)].mean(axis=0)

    def _update_follow_cam(self) -> None:
        if not self.follow_cam:
            return
        camera = self.scene.camera
        if not isinstance(camera, ViewerCamera):
            return
        center = self._current_center()
        if center is None:
            return
        if self._follow_center is None:
            offset = camera.position - camera.target
            camera.target = center
            camera.position = center + offset
        else:
            delta = center - self._follow_center
            camera.position = camera.position + delta
            camera.target = camera.target + delta
        self._follow_center = center


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive streaming text->motion studio (aitviewer)."
    )
    p.add_argument("--config", default="configs/generator/final100m_fsq8x1024.yaml")
    p.add_argument("--ckpt", default="checkpoints/generator/generator_transformer_100m_val163.pt")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    p.add_argument("--backbone", default="transformer", choices=list(BACKBONES))
    p.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="SMPL-X model dir; if missing the studio shows the skeleton only.",
    )
    p.add_argument("--steps", type=int, default=49)
    p.add_argument("--cfg_scale", type=float, default=6.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(2026, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] connecting to the motion service (backbone={args.backbone})...")
    client = connect_or_spawn(
        args.config, args.ckpt, args.tokenizer_ckpt, args.backbone, args.host, args.port
    )  # separate process owns the model: generation never fights the render loop for the GIL
    model_dir = args.model_dir if args.model_dir and Path(args.model_dir).exists() else ""
    if not model_dir:
        print("[load] SMPL-X model dir not found -> skeleton-only studio")

    viewer = StreamingStudioViewer(client=client, model_dir=model_dir, device=device, args=args)
    viewer.run()


if __name__ == "__main__":
    main()
