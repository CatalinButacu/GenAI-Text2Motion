"""Interactive streaming text->motion studio, built by extending the aitviewer ``Viewer``.

Layout (a fixed tiled studio, immune to a stale ``imgui.ini``):
  * LEFT-top      -- Scene editor (aitviewer hierarchy + inspector), pinned.
  * LEFT-bottom   -- Environment: background, ambient, per-light controls, shadows, floor, origin.
  * RIGHT-top     -- Generation: runtime model/tokenizer selection, sampling params, avatar
                     (gender + shape sliders, applied to the on-screen body in real time).
  * RIGHT-bottom  -- Playback: play/pause, frame scrubber, speed, camera follow, render stats.
  * BOTTOM-center -- prompt bar (chat-style) with motion + body-fit progress bars.
  * CENTER        -- the avatar, animating LIVE as the bounded-state decoder emits frames.

All custom panels are injected through ``self.gui_controls`` so their imgui calls run inside the
viewer's frame (never outside -> no ``WithinFrameScope`` crash). All scene-node mutations happen in
``on_render`` (main thread), fed by the background threads (generation, avatar re-mesh, model
loading) through a lock-guarded handoff.
"""

from __future__ import annotations

import argparse
import threading
import traceback
from array import array
from dataclasses import dataclass, replace
from pathlib import Path

import imgui
import numpy as np
import torch
import yaml
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import ViewerCamera
from aitviewer.viewer import Viewer

from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.render.joints2smpl import (
    FitConfig,
    build_smplx_model,
    fit_smplx_to_joints,
    mesh_from_params,
    rest_pose_body,
)
from text2motion.render.studio import build_skeleton_seq, recover_skeleton
from text2motion.shared.config import load_config
from text2motion.shared.seed import seed_everything
from text2motion.stream.decode import StreamingMotionDecoder

SKIN_COLOR = (0.86, 0.72, 0.61, 1.0)  # SMPL-X body tone
SKY_COLOR = (240 / 255, 182 / 255, 182 / 255, 1.0)  # default background: soft pink
GENDERS = ("neutral", "male", "female")
BACKBONES = ("transformer", "mamba")
N_BETAS = 10  # SMPL-X shape dims exposed as avatar-dimension sliders
ROOT = Path(__file__).resolve().parents[2]
# Local copy (data/ is repo-root-ignored; license-gated, never commit).
DEFAULT_MODEL_DIR = str(ROOT / "data" / "smplx_models")


def load_pipeline(config: str, ckpt: str, tokenizer_ckpt: str, backbone: str, device: str):
    """Load tokenizer + generator + CLIP text encoder + normalisation stats onto ``device``."""
    cfg = load_config(config)

    tok = ResidualFsqTokenizer(cfg.tokenizer)
    tok.load_state_dict(torch.load(tokenizer_ckpt, map_location="cpu"))
    tok.to(device).eval()

    n_layers = cfg.generator.mamba_n_layers if backbone == "mamba" else cfg.generator.n_layers
    gen_cfg = replace(
        cfg.generator,
        backbone=backbone,
        n_layers=n_layers,
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=tok.codebook_size,
        use_kernel=False,
    )
    gen = MotionGenerator(gen_cfg).to(device).eval()
    state = torch.load(ckpt, map_location="cpu")
    gen.load_state_dict(
        state["generator"] if isinstance(state, dict) and "generator" in state else state
    )

    te = CLIPTextEncoder(cfg.text_encoder).to(device).eval()

    out = Path(cfg.paths.hml3d_out_dir)
    mean = torch.from_numpy(np.load(out / "Mean.npy").astype("float32")).to(device)
    std = torch.from_numpy(np.load(out / "Std.npy").astype("float32")).to(device)
    return tok, gen, te, mean, std


@dataclass(frozen=True)
class ModelEntry:
    """One loadable model bundle: a checkpoint pinned to the EXACT config + tokenizer it was
    trained with (from configs/demo_models.yaml) -- no free mixing at inference time."""

    label: str
    config: str
    ckpt: str
    tokenizer_ckpt: str
    backbone: str


MODEL_REGISTRY = ROOT / "configs" / "demo_models.yaml"


def load_model_registry(launch: ModelEntry) -> tuple[list[ModelEntry], int]:
    """Registry entries + the index matching the launch args (which are appended if unknown)."""
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
    """aitviewer studio that streams a text prompt into a live-animating avatar."""

    def __init__(self, pipeline, model_dir: str, device: str, args: argparse.Namespace) -> None:
        super().__init__()
        self.tok, self.gen, self.te, self.mean, self.std = pipeline
        self.device = device
        self.model_dir = model_dir  # "" -> SMPL-X unavailable: skeleton-only mode

        # --- sampling state (driven by the RIGHT panel) ---
        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.cfg_scale = float(args.cfg_scale)
        self.steps = int(args.steps)

        # --- avatar personalization (applied to the on-screen body in real time) ---
        self.gender_idx = 0
        self.fit_body = bool(model_dir)
        self.skin_color = SKIN_COLOR
        self.user_betas = np.zeros(N_BETAS, dtype=np.float32)  # offsets on the fitted shape

        # --- runtime model selection (registry-pinned bundles only) ---
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

        # --- chat state ---
        self.prompt_text = ""
        self.status = "ready -- describe a motion below and press Enter"
        self.generating = False
        self.history: list[str] = []

        # --- cross-thread handoff (workers produce, on_render consumes) ---
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

        # --- SMPL-X params of the current motion, accumulated per fitted chunk; feeding
        # mesh_from_params with a NEW gender/betas re-dresses the avatar without re-fitting ---
        self._fit_orient: list[np.ndarray] = []
        self._fit_pose: list[np.ndarray] = []
        self._fit_transl: list[np.ndarray] = []
        self._fit_betas: np.ndarray | None = None
        self._remesh_version = 0
        self._remesh_done_version = 0
        self._remesh_running = False
        self._remesh_after_gen = False
        self._remesh_note = ""

        # --- progress (prompt bar) ---
        self._gen_progress = 0.0
        self._fit_progress: float | None = None

        # --- scene nodes we own ---
        self._rest_node = None
        self._motion_node = None
        self._body_node = None

        # --- follow-cam: keep the avatar as the camera pivot as its root translates ---
        self.follow_cam = True
        self._follow_center: np.ndarray | None = None

        # deterministic tiled layout; pinned, so a stale imgui.ini can't scatter the panels
        self._flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE
        self._styled = False
        self.playback_fps = float(args.fps)
        self.run_animations = False
        self.scene.background_color = SKY_COLOR

        self._install_panels()
        self._add_rest_pose()

    # ------------------------------------------------------------------ setup

    def _install_panels(self) -> None:
        """Add the studio panels; gui_scene/gui_playback are overridden in place (same dict keys)."""
        self.gui_controls["environment"] = self.gui_environment
        self.gui_controls["params"] = self.gui_params
        self.gui_controls["prompt"] = self.gui_prompt

    def _add_rest_pose(self) -> None:
        """Show a standing SMPL-X body at launch so the studio is never empty (if models are present)."""
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

    # --------------------------------------------------------------- geometry

    def _rects(self):
        """Panel rectangles (x, y, w, h) for the tiled layout -- pure percentages of imgui's OWN
        coordinate space. ``self.window_size`` is physical pixels while imgui positions windows in
        logical units; under Windows display scaling (laptop at 125/150%) the two differ, so any
        pixel-based rect overflows the screen. ``io.display_size`` is correct on every monitor."""
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

    # ------------------------------------------------------------------ style

    def _apply_style(self) -> None:
        """One-time imgui restyle: rounded, dark-slate panels with a blue accent."""
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

    # ------------------------------------------------------------------ panels

    def gui_scene(self) -> None:
        """LEFT-top: the default aitviewer Editor (hierarchy + inspector), pinned."""
        self._apply_style()
        editor, *_ = self._rects()
        self._dock(editor)
        expanded, _ = imgui.begin("Scene", False, self._flags)
        if expanded:
            self.scene.gui_editor(imgui, self.viewports, self.viewport_mode)
        imgui.end()

    def gui_environment(self) -> None:
        """LEFT-bottom: everything about the stage -- background, lights, shadows, floor."""
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
        """RIGHT-top: runtime model selection, sampling params, and the live avatar editor."""
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
        seconds = self.steps * self.tok.cfg.downsample / self.playback_fps
        self._dim_text(
            f"= {self.steps * self.tok.cfg.downsample} frames (~{seconds:.1f}s @"
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
        """One SMPL-X shape slider; the re-mesh fires on release, not on every drag pixel."""
        changed, value = imgui.slider_float(label, float(self.user_betas[i]), -3.0, 3.0)
        if changed:
            self.user_betas[i] = value
        if imgui.is_item_deactivated_after_edit():
            self._on_avatar_changed()

    def gui_playback(self) -> None:
        """RIGHT-bottom: transport + camera + render stats (replaces the floating default)."""
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
        """BOTTOM-center: chat-style prompt bar with live motion + body-fit progress bars."""
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
        imgui.pop_item_width()
        imgui.same_line()
        if self.generating:
            if imgui.button("Stop", width=button_w):
                self._cancel = True
        elif imgui.button("Generate", width=button_w) or enter:
            self.submit_prompt()

        if self.generating:
            avail = imgui.get_content_region_available()[0]
            done = int(self._gen_progress * self.steps * self.tok.cfg.downsample)
            total = self.steps * self.tok.cfg.downsample
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

    # -------------------------------------------------------------- generation

    def submit_prompt(self) -> None:
        """Kick off a background generation for the current prompt (main-thread, from gui_prompt)."""
        prompt = self.prompt_text.strip()
        if not prompt or self.generating:
            return
        self.generating = True
        self._cancel = False
        self._gen_progress = 0.0
        self._fit_progress = None
        self.status = f'generating: "{prompt}"'
        self.history.append(prompt)
        fit_body = self.fit_body and bool(self.model_dir)
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "fit_body": fit_body,
            "downsample": self.tok.cfg.downsample,
        }
        self._current_fit_body = fit_body
        # the previous avatar/skeleton stays on screen (frozen) until the first new chunk streams
        # in; _append_chunks then swaps the display in one shot -- no empty-scene flash.
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

    def _worker(self, prompt: str, params: dict) -> None:
        """Background thread: stream tokens -> joint chunks, fitting a SMPL-X body PER CHUNK (not
        once at the end) so the avatar visibly grows/animates while generation is still running."""
        n_frames = 0
        total_frames = params["steps"] * params["downsample"]
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        try:
            with torch.no_grad():
                emb = self.te([prompt]).to(self.device)
                decoder = StreamingMotionDecoder(self.tok, self.mean, self.std)
                token_iter = self.gen.stream(
                    emb,
                    params["steps"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    cfg_scale=params["cfg_scale"],
                    stop_at_end=False,
                )
                for chunk in decoder.stream_tokens(token_iter):
                    if self._cancel:
                        break
                    joints = recover_skeleton(chunk.squeeze(0).cpu().numpy())  # (t, 22, 3)
                    n_frames += len(joints)
                    self._gen_progress = min(1.0, n_frames / total_frames)
                    # the skeleton streams IMMEDIATELY in both modes (decode is ms-fast); the
                    # SMPL-X body materializes behind it as each chunk's slow fit completes
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
        # gender/shape are honoured live per chunk; a full-clip remesh is only needed when
        # options changed MID-generation (earlier chunks were fitted with the old ones)
        if params["fit_body"] and self._remesh_after_gen and self._fit_orient:
            self._remesh_after_gen = False
            self._schedule_remesh()

    def _fit_chunk(
        self,
        joints: np.ndarray,
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        n_frames_so_far: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Fit SMPL-X to a single streamed chunk, warm-started from the previous chunk (skips the
        slow yaw search + re-optimizes only a handful of iterations), and hand the resulting mesh
        chunk off to the render thread. Gender and shape offsets are read LIVE, so changing them
        mid-generation shows up from the next chunk on (the end-of-run remesh unifies the clip).
        Returns the new warm-start (or the old one, unchanged, if this chunk's fit fails -- one
        bad chunk shouldn't derail the rest of the generation)."""
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
        """The chunk's display mesh with the user's shape sliders applied on top of the fitted
        betas -- one no-grad forward of the already-built model, so the streaming body honours
        height/build in real time instead of only after the end-of-run remesh."""
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

    # ------------------------------------------------- live avatar re-dressing

    def _on_avatar_changed(self) -> None:
        """Gender/shape edits re-dress the CURRENT body in real time (no re-generation): the stored
        per-chunk SMPL-X params are re-forwarded with the new gender/betas via mesh_from_params."""
        if not self.model_dir or not self.fit_body:
            return
        if self.generating:
            self._remesh_after_gen = True
            self._remesh_note = "new chunks use the new avatar; full clip refreshes when done"
            return
        self._schedule_remesh()

    def _on_fit_body_toggled(self) -> None:
        """Make the checkbox act on the CURRENT display, not just the next generation: OFF hides
        the body (the motion falls back to its skeleton); ON re-dresses instantly from stored fit
        params, retro-fits a skeleton-only motion, or restores the rest pose."""
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
        """Fit SMPL-X to an already-generated motion, chunk by chunk with warm starts -- same
        machinery and same growing-body visuals as a live streaming fit."""
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
        """Background thread: re-forward SMPL-X with the current gender + shape offsets. Loops
        until the requested version is stable, so a burst of slider edits collapses into the
        latest one instead of queueing a mesh per pixel of drag."""
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

    # --------------------------------------------------------- model reloading

    def _start_model_load(self) -> None:
        entry = self._models[self._model_idx]
        self._loading_model = True
        self.load_status = f"loading {entry.label}..."
        threading.Thread(target=self._load_worker, args=(entry,), daemon=True).start()

    def _load_worker(self, entry: ModelEntry) -> None:
        """Background thread: build the new pipeline fully, then hand it to on_render to swap."""
        try:
            pipeline = load_pipeline(
                entry.config, entry.ckpt, entry.tokenizer_ckpt, entry.backbone, self.device
            )
            with self.lock:
                self._pending_pipeline = (pipeline, entry.backbone, entry.label)
        except Exception as exc:  # noqa: BLE001 - a missing/mismatched bundle must not crash the studio
            traceback.print_exc()
            self.load_status = f"load failed: {type(exc).__name__}: {exc}"
            self._loading_model = False

    # --------------------------------------------------------------- rendering

    def on_render(self, time, frame_time, **kwargs) -> None:
        """Apply pending scene mutations on the main thread, then render + draw the GUI.

        Playback is the base class's looping advance (``current_frame_id + frames) % n_frames``),
        deliberately left in charge DURING generation too: every motion plays as an infinite loop,
        wrapping over the partial clip while it still grows and over the full clip afterwards.
        """
        self._consume()
        self._sync_display_nodes()
        self._update_follow_cam()
        super().on_render(time, frame_time, **kwargs)

    def _sync_display_nodes(self) -> None:
        """Exactly ONE figure per rendered frame while skeleton and body coexist: the SMPL-X body
        wherever its fit already reached, the skeleton only for the not-yet-fitted tail. Without
        this, a play head past the body frontier shows the clamped (frozen) mesh AND the moving
        skeleton at once -- two disconnected characters."""
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
            pipeline = self._pending_pipeline
            self._pending_pipeline = None

        if pipeline is not None:
            (self.tok, self.gen, self.te, self.mean, self.std), self.backbone, name = pipeline
            self.load_status = f"active: {name}"
            self._loading_model = False
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
            try:
                self.scene.remove(node)
            except Exception:  # noqa: BLE001
                pass
            setattr(self, attr, None)

    def _append_chunks(self, chunks: list[np.ndarray]) -> None:
        """Show newly-streamed joint frames as a live skeleton, in BOTH modes: decode is ms-fast,
        so the skeleton is what makes each generated batch visible the moment it exists. In
        fit-body mode the SMPL-X body materializes behind it, and ``_sync_display_nodes`` shows
        exactly one figure per frame (body where fitted, skeleton for the tail). The first chunk
        retires the previous avatar in the same render pass -- no empty-scene flash."""
        new = np.concatenate(chunks, axis=0)
        self._accum_joints = (
            new if self._accum_joints is None else np.concatenate([self._accum_joints, new], axis=0)
        )
        self._remove("_motion_node")
        self._motion_node = build_skeleton_seq(self._accum_joints)
        self._motion_node.name = "Motion (streaming)"
        self.scene.add(self._motion_node)
        if not self._started:
            self._started = True
            self._remove("_body_node")
            self._remove("_rest_node")
            self._follow_center = None  # re-anchor the follow-cam on the new clip
            self.scene.current_frame_id = 0
            self.toggle_animation(True)
            self._need_reset_view = True

    def _append_body_chunks(self, chunks: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Grow the SMPL-X avatar live, one fitted chunk at a time, trailing the streaming
        skeleton; the play head keeps following the skeleton frontier, so the body catches up
        without yanking playback backwards."""
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
        """Swap the re-dressed avatar in-place, preserving the play head (main thread only)."""
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
        """World-space centroid of the avatar at the current frame (the camera pivot)."""
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
        """Keep the avatar centred by translating the camera rig with its per-frame motion,
        preserving the user's current orbit/zoom (root translation is what walked it off-screen)."""
        if not self.follow_cam:
            return
        camera = self.scene.camera
        if not isinstance(camera, ViewerCamera):
            return
        center = self._current_center()
        if center is None:
            return
        if self._follow_center is None:
            # first frame of a new clip: move the pivot onto the avatar, keep the viewing angle
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
    return p


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(2026, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] pipeline on {device} (backbone={args.backbone})...")
    pipeline = load_pipeline(args.config, args.ckpt, args.tokenizer_ckpt, args.backbone, device)
    model_dir = args.model_dir if args.model_dir and Path(args.model_dir).exists() else ""
    if not model_dir:
        print("[load] SMPL-X model dir not found -> skeleton-only studio")

    viewer = StreamingStudioViewer(pipeline=pipeline, model_dir=model_dir, device=device, args=args)
    viewer.run()


if __name__ == "__main__":
    main()
