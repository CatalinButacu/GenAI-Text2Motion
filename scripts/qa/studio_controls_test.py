from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import imgui
import numpy as np
from studio_human_test import HumanDriver, build_studio_viewer

from text2motion.app.config import load_dataclass, load_service_config, service_policy
from text2motion.streaming.protocol import MotionStreamProtocol, motion_service_client
from text2motion.studio.config import StudioConfig

TRACKED_WIDGETS = (
    "button",
    "small_button",
    "checkbox",
    "radio_button",
    "slider_float",
    "slider_int",
    "drag_float",
    "combo",
    "color_edit3",
    "color_edit4",
    "input_text_with_hint",
    "selectable",
    "collapsing_header",
    "tree_node",
)


@dataclass
class WidgetRect:
    x0: float
    y0: float
    x1: float
    y1: float
    clip_y0: float
    clip_y1: float
    label_width: float
    line_height: float
    item_spacing_y: float
    window_padding_y: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def frame_x1(self) -> float:
        return self.x1 - self.label_width


def visible_label(label: str) -> str:
    return label.split("##")[0]


class WidgetTracker:
    def __init__(self) -> None:
        self.rects: dict[str, WidgetRect] = {}
        self.kinds: dict[str, str] = {}
        self.fired: set[str] = set()
        self.originals = {name: getattr(imgui, name) for name in TRACKED_WIDGETS}
        for name, original in self.originals.items():
            setattr(imgui, name, self.__wrap(name, original))

    def __wrap(self, kind, original):
        def tracked(label, *args, **kwargs):
            result = original(label, *args, **kwargs)
            self.kinds[label] = kind
            if kind in ("button", "small_button") and result:
                self.fired.add(label)
            x0, y0 = imgui.get_item_rect_min()
            x1, y1 = imgui.get_item_rect_max()
            window_y = imgui.get_window_position()[1]
            window_h = imgui.get_window_size()[1]
            style = imgui.get_style()
            text = visible_label(label)
            label_width = 0.0
            if text:
                label_width = imgui.calc_text_size(text)[0] + style.item_inner_spacing[0]
            self.rects[label] = WidgetRect(
                x0,
                y0,
                x1,
                y1,
                window_y,
                window_y + window_h,
                label_width,
                imgui.get_text_line_height(),
                style.item_spacing[1],
                style.window_padding[1],
            )
            return result

        return tracked

    def restore(self) -> None:
        for name, original in self.originals.items():
            setattr(imgui, name, original)


class Pointer:
    def __init__(self, driver: HumanDriver, tracker: WidgetTracker) -> None:
        self.driver = driver
        self.tracker = tracker
        self.viewer = driver.viewer

    @property
    def left(self):
        return self.viewer.wnd.mouse.left

    def rect(self, label: str) -> WidgetRect:
        self.driver.frame()
        if label not in self.tracker.rects:
            raise LookupError(f"widget {label!r} was not drawn this frame")
        return self.tracker.rects[label]

    def reveal(self, label: str) -> WidgetRect:
        rect = self.rect(label)
        for _ in range(40):
            if rect.y0 >= rect.clip_y0 + 20 and rect.y1 <= rect.clip_y1 - 8:
                return rect
            direction = -1.0 if rect.y1 > rect.clip_y1 - 8 else 1.0
            x, _ = rect.center
            self.viewer.mouse_position_event(int(x), int((rect.clip_y0 + rect.clip_y1) / 2), 0, 0)
            self.viewer.mouse_scroll_event(0.0, direction)
            self.driver.frame()
            rect = self.rect(label)
        return rect

    def move(self, x: float, y: float) -> None:
        self.viewer.mouse_position_event(int(x), int(y), 0, 0)
        self.driver.frame()

    def click_at(self, x: float, y: float) -> None:
        self.move(x, y)
        self.viewer.mouse_press_event(int(x), int(y), self.left)
        self.driver.frame()
        self.viewer.mouse_release_event(int(x), int(y), self.left)
        self.driver.frame()
        self.driver.frame()
        self.driver.linger()

    def settled(self, label: str) -> WidgetRect:
        rect = self.reveal(label)
        for _ in range(30):
            self.move(*rect.center)
            fresh = self.rect(label)
            if abs(fresh.y0 - rect.y0) < 0.5 and abs(fresh.x0 - rect.x0) < 0.5:
                return fresh
            rect = fresh
        return rect

    def click(self, label: str) -> None:
        rect = self.settled(label)
        x, y = rect.center
        self.tracker.fired.discard(label)
        self.click_at(x, y)
        is_button = self.tracker.kinds.get(label) in ("button", "small_button")
        if is_button and label not in self.tracker.fired:
            self.driver.check(
                f"click on '{label}' registers",
                False,
                f"clicked ({x:.0f},{y:.0f}) rect ({rect.x0:.0f},{rect.y0:.0f})-({rect.x1:.0f},{rect.y1:.0f}) "
                f"window clip y {rect.clip_y0:.0f}-{rect.clip_y1:.0f}",
            )

    def drag(self, x0: float, y0: float, x1: float, y1: float, steps: int = 10) -> None:
        self.move(x0, y0)
        self.viewer.mouse_press_event(int(x0), int(y0), self.left)
        self.driver.frame()
        previous_x, previous_y = x0, y0
        for step in range(1, steps + 1):
            x = x0 + (x1 - x0) * step / steps
            y = y0 + (y1 - y0) * step / steps
            self.viewer.mouse_drag_event(int(x), int(y), int(x - previous_x), int(y - previous_y))
            self.driver.frame()
            previous_x, previous_y = x, y
        self.viewer.mouse_release_event(int(x1), int(y1), self.left)
        self.driver.frame()
        self.driver.frame()

    def set_slider(self, label: str, fraction: float) -> None:
        rect = self.reveal(label)
        x = rect.x0 + (rect.frame_x1 - rect.x0) * fraction
        self.click_at(x, (rect.y0 + rect.y1) / 2)

    def drag_first_field(self, label: str, pixels: float) -> None:
        rect = self.reveal(label)
        x = rect.x0 + 6
        y = (rect.y0 + rect.y1) / 2
        self.drag(x, y, x + pixels, y)

    def pick_combo_item(self, label: str, index: int) -> None:
        rect = self.reveal(label)
        self.click(label)
        row = rect.line_height + rect.item_spacing_y
        y = rect.y1 + rect.window_padding_y + index * row + rect.line_height / 2
        self.click_at(rect.x0 + 24, y)

    def type_into(self, label: str, text: str) -> None:
        self.click(label)
        for character in text:
            self.viewer.unicode_char_entered(character)
            self.driver.frame()

    def press_enter(self) -> None:
        keys = self.viewer.wnd.keys
        self.viewer.key_event(keys.ENTER, keys.ACTION_PRESS, None)
        self.driver.frame()
        self.viewer.key_event(keys.ENTER, keys.ACTION_RELEASE, None)
        self.driver.frame()


def environment_controls(driver: HumanDriver, pointer: Pointer) -> None:
    driver.scenario = "environment_panel"
    viewer = driver.viewer
    scene = viewer.scene

    before = tuple(scene.background_color)
    pointer.drag_first_field("background", -60)
    driver.check(
        "background colour edit changes the background",
        tuple(scene.background_color) != before,
        f"{before} -> {tuple(scene.background_color)}",
    )

    pointer.set_slider("ambient", 0.25)
    driver.check(
        "ambient slider sets ambient strength",
        abs(scene.ambient_strength - 1.0) < 0.35,
        f"{scene.ambient_strength:.2f} (expected ~1.0)",
    )

    for mode in ["dark", "diffuse", "default"]:
        pointer.click(mode)
        driver.check(
            f"mood button '{mode}' sets the light mode", scene.light_mode == mode, scene.light_mode
        )

    shadows = viewer.shadows_enabled
    pointer.click("shadows [S]")
    driver.check(
        "shadows checkbox toggles shadows",
        viewer.shadows_enabled != shadows,
        f"{shadows} -> {viewer.shadows_enabled}",
    )
    pointer.click("shadows [S]")

    light = scene.lights[0]
    pointer.set_slider("strength##l0", 0.5)
    driver.check(
        "light strength slider",
        abs(light.strength - 5.0) < 1.0,
        f"{light.strength:.2f} (expected ~5)",
    )
    pointer.set_slider("elevation##l0", 0.75)
    driver.check(
        "light elevation slider",
        abs(light.elevation - 45.0) < 12.0,
        f"{light.elevation:.1f} (expected ~45)",
    )
    pointer.set_slider("azimuth##l0", 0.25)
    driver.check(
        "light azimuth slider",
        abs(light.azimuth - 90.0) < 25.0,
        f"{light.azimuth:.1f} (expected ~90)",
    )
    before = tuple(light.light_color[:3])
    pointer.drag_first_field("color##l0", -60)
    driver.check(
        "light colour edit",
        tuple(light.light_color[:3]) != before,
        f"{before} -> {tuple(light.light_color[:3])}",
    )
    cast = light.shadow_enabled
    pointer.click("cast shadows##l0")
    driver.check(
        "light cast-shadows checkbox",
        light.shadow_enabled != cast,
        f"{cast} -> {light.shadow_enabled}",
    )
    pointer.click("cast shadows##l0")

    floor = scene.floor
    enabled = floor.enabled
    pointer.click("show floor")
    driver.check("show floor checkbox", floor.enabled != enabled, f"{enabled} -> {floor.enabled}")
    pointer.click("show floor")
    before = tuple(float(v) for v in floor.c1[:3])
    pointer.drag_first_field("tile a", -60)
    driver.check("floor tile a colour", tuple(float(v) for v in floor.c1[:3]) != before, "")
    before = tuple(float(v) for v in floor.c2[:3])
    pointer.drag_first_field("tile b", -60)
    driver.check("floor tile b colour", tuple(float(v) for v in floor.c2[:3]) != before, "")
    origin = scene.origin.enabled
    pointer.click("origin axes")
    driver.check(
        "origin axes checkbox",
        scene.origin.enabled != origin,
        f"{origin} -> {scene.origin.enabled}",
    )
    pointer.click("origin axes")


def sampling_controls(driver: HumanDriver, pointer: Pointer) -> None:
    driver.scenario = "sampling_panel"
    state = driver.viewer.state
    pointer.set_slider("temperature", 0.5)
    driver.check(
        "temperature slider",
        abs(state.temperature - 1.05) < 0.2,
        f"{state.temperature:.2f} (expected ~1.05)",
    )
    pointer.set_slider("top_p", 0.8)
    driver.check(
        "top_p slider", abs(state.top_p - 0.82) < 0.1, f"{state.top_p:.2f} (expected ~0.82)"
    )
    pointer.set_slider("cfg_scale", 0.5)
    driver.check(
        "cfg_scale slider",
        abs(state.cfg_scale - 6.5) < 1.2,
        f"{state.cfg_scale:.2f} (expected ~6.5)",
    )

    if state.max_steps == 0:
        state.steps = 0
        pointer.click("run forever (bounded state)")
        driver.check(
            "unticking 'run forever' gives a finite budget",
            state.steps == state.default_steps,
            f"steps {state.steps}",
        )
        pointer.set_slider("steps (tokens)", 0.25)
        upper = state.max_steps or state.window_steps
        expected = 8 + 0.25 * (upper - 8)
        driver.check(
            "steps slider",
            abs(state.steps - expected) < 0.15 * upper,
            f"{state.steps} (expected ~{expected:.0f})",
        )
        pointer.click("run forever (bounded state)")
        driver.check(
            "ticking 'run forever' makes the budget unbounded",
            state.steps == 0,
            f"steps {state.steps}",
        )
    state.temperature, state.top_p, state.cfg_scale = 1.0, 0.9, 6.0


def prompt_playback(driver: HumanDriver, pointer: Pointer) -> None:
    driver.scenario = "prompt_playback"
    viewer = driver.viewer
    state = viewer.state
    state.steps = 0
    state.prompt_text = ""
    pointer.type_into("##prompt", "a person walks")
    driver.check(
        "typing into the prompt box fills the prompt",
        state.prompt_text == "a person walks",
        repr(state.prompt_text),
    )
    pointer.press_enter()
    driver.check("Enter in the prompt box starts a generation", state.generating, state.status)
    driver.check(
        "the Stop button replaces Generate while generating",
        driver.wait_until(lambda: "Stop" in pointer.tracker.rects, 2.0),
        "",
    )
    driver.wait(1.5)
    pointer.click("Stop")
    driver.check(
        "Stop button ends the generation",
        driver.wait_until(lambda: not state.generating, 5.0),
        state.status,
    )
    driver.check("the stopped take is reported as stopped", "stopped" in state.status, state.status)
    state.steps = 24
    pointer.click("Generate")
    driver.check("Generate button starts a generation", state.generating, state.status)
    driver.wait_until(lambda: not state.generating, 30.0)
    driver.wait_until(
        lambda: (
            state.nodes.motion is not None
            and state.stream_buffers.body_frames == len(state.nodes.motion.joint_positions)
        ),
        20.0,
    )

    skin = state.show_skin
    pointer.click("hide skin (H)" if skin else "show skin (H)")
    driver.check(
        "hide/show skin button toggles the skin",
        state.show_skin != skin,
        f"{skin} -> {state.show_skin}",
    )
    driver.frame()
    relabelled = ("show skin (H)" if skin else "hide skin (H)") in pointer.tracker.rects
    driver.check("the skin button relabels itself", relabelled, "")
    pointer.click("show skin (H)" if skin else "hide skin (H)")

    live = state.live.enabled
    pointer.click("live (regenerate as you type)")
    driver.check(
        "live checkbox toggles live mode",
        state.live.enabled != live,
        f"{live} -> {state.live.enabled}",
    )
    pointer.click("live (regenerate as you type)")
    driver.wait_until(lambda: not state.generating, 30.0)

    running = viewer.run_animations
    pointer.click("play [space]")
    driver.check(
        "play checkbox pauses/resumes playback",
        viewer.run_animations != running,
        f"{running} -> {viewer.run_animations}",
    )
    if viewer.run_animations:
        pointer.click("play [space]")
    n_frames = viewer.scene.n_frames
    pointer.set_slider("##frame", 0.5)
    driver.check(
        "frame scrubber moves the playhead",
        abs(viewer.scene.current_frame_id - (n_frames - 1) / 2) <= max(3, 0.08 * n_frames),
        f"frame {viewer.scene.current_frame_id} of {n_frames} (expected ~{(n_frames - 1) / 2:.0f})",
    )
    pointer.click("play [space]")

    fps = viewer.playback_fps
    rect = pointer.reveal("speed (fps)")
    pointer.drag(rect.x0 + 20, (rect.y0 + rect.y1) / 2, rect.x0 + 70, (rect.y0 + rect.y1) / 2)
    driver.check(
        "speed drag changes playback fps",
        viewer.playback_fps > fps,
        f"{fps:.1f} -> {viewer.playback_fps:.1f}",
    )
    viewer.playback_fps = fps

    follow = state.follow_cam
    pointer.click("camera follows avatar")
    driver.check(
        "follow-camera checkbox", state.follow_cam != follow, f"{follow} -> {state.follow_cam}"
    )
    pointer.click("camera follows avatar")
    camera = viewer.scene.camera
    camera.position = camera.position + np.array([3.0, 2.0, 3.0])
    moved = np.array(camera.position, dtype=float)
    state.follow_cam = False
    pointer.click("center view")
    driver.wait(0.3)
    driver.check(
        "center view button re-frames the camera",
        not np.allclose(np.array(camera.position, dtype=float), moved),
        f"camera {np.round(moved, 2)} -> {np.round(np.array(camera.position, dtype=float), 2)}",
    )
    state.follow_cam = True


def avatar_controls(driver: HumanDriver, pointer: Pointer) -> None:
    driver.scenario = "avatar_panel"
    state = driver.viewer.state
    driver.wait_until(lambda: not state.generating and not state.smplx_fit.running, 20.0)

    pointer.click("avatar skin (SMPL-X body)")
    driver.check(
        "avatar skin checkbox turns the body off",
        not state.fit_body and state.nodes.body is None,
        f"fit_body {state.fit_body}, body node {state.nodes.body is not None}",
    )
    pointer.click("avatar skin (SMPL-X body)")
    driver.check(
        "avatar skin checkbox brings the body back",
        state.fit_body
        and driver.wait_until(lambda: state.nodes.body is not None and not state.generating, 20.0),
        f"fit_body {state.fit_body}, body node {state.nodes.body is not None}",
    )

    genders = state.config.genders
    for index, name in enumerate(genders):
        if index == state.gender_idx:
            continue
        version = state.smplx_fit.version
        pointer.click(name)
        driver.check(f"gender radio '{name}' selects it", state.gender == name, state.gender)
        driver.check(
            f"gender radio '{name}' rebuilds the avatar", state.smplx_fit.version > version, ""
        )
        driver.wait_until(lambda: not state.smplx_fit.running, 20.0)
    pointer.click(genders[0])
    driver.wait_until(lambda: not state.smplx_fit.running, 20.0)

    before = tuple(state.skin_color[:3])
    pointer.drag_first_field("skin", -60)
    driver.check(
        "skin colour edit changes the skin",
        tuple(state.skin_color[:3]) != before,
        f"{before} -> {tuple(state.skin_color[:3])}",
    )
    body = state.nodes.body
    driver.check(
        "skin colour reaches the body mesh",
        body is not None and np.allclose(np.array(body.color[:3]), np.array(state.skin_color[:3])),
        "",
    )

    for index, label in [(0, "height"), (1, "build")]:
        version = state.smplx_fit.version
        pointer.set_slider(label, 0.8)
        value = float(state.user_betas[index])
        driver.check(
            f"{label} slider sets the shape", abs(value - 1.8) < 0.7, f"{value:.2f} (expected ~1.8)"
        )
        driver.check(
            f"{label} slider rebuilds the avatar on release", state.smplx_fit.version > version, ""
        )
        driver.wait_until(lambda: not state.smplx_fit.running, 20.0)

    pointer.click("more shape dimensions")
    driver.frame()
    driver.check(
        "'more shape dimensions' reveals extra sliders", "shape 2" in pointer.tracker.rects, ""
    )
    if "shape 2" in pointer.tracker.rects:
        pointer.set_slider("shape 2", 0.2)
        driver.check(
            "shape 2 slider", float(state.user_betas[2]) < -1.0, f"{state.user_betas[2]:.2f}"
        )
    pointer.click("more shape dimensions")

    pointer.click("reset shape")
    driver.check(
        "reset shape zeroes the shape",
        not state.user_betas.any(),
        str(np.round(state.user_betas, 2)),
    )
    driver.wait_until(lambda: not state.smplx_fit.running and not state.generating, 20.0)
    version = state.smplx_fit.version
    pointer.click("refresh avatar")
    driver.check("refresh avatar rebuilds the avatar", state.smplx_fit.version > version, "")
    driver.wait_until(lambda: not state.smplx_fit.running, 20.0)

    if state.history:
        state.prompt_text = ""
        pointer.click("prompt history")
        driver.frame()
        label = next((key for key in pointer.tracker.rects if key.endswith("##hist0")), None)
        driver.check("prompt history lists past prompts", label is not None, "")
        if label is not None:
            pointer.click(label)
            driver.check(
                "clicking a past prompt puts it back in the prompt box",
                state.prompt_text == state.history[-1],
                repr(state.prompt_text),
            )
        pointer.click("prompt history")


def model_controls(driver: HumanDriver, pointer: Pointer) -> None:
    driver.scenario = "model_panel"
    state = driver.viewer.state
    labels = [entry.key for entry in state.models]
    transformer = next(
        i for i, key in enumerate(labels) if key.startswith("transformer:") and "thesis" in key
    )
    mamba = next(i for i, key in enumerate(labels) if key.startswith("mamba:") and "thesis" in key)
    pointer.pick_combo_item("model", transformer)
    driver.check(
        "model combo selects an entry",
        state.model_idx == transformer,
        f"index {state.model_idx}, wanted {transformer}",
    )
    pointer.click("Load model")
    loaded = driver.wait_until(
        lambda: not state.loading_model and "loading" not in state.load_status, 120.0
    )
    driver.check(
        "Load model loads the selected model",
        loaded and "transformer" in state.load_status,
        state.load_status,
    )
    pointer.pick_combo_item("model", mamba)
    pointer.click("Load model")
    driver.wait_until(lambda: not state.loading_model and "loading" not in state.load_status, 120.0)
    driver.check("Load model switches back", "mamba" in state.load_status, state.load_status)
    pointer.click("Reload settings + models (R)")
    driver.check("Reload settings button reloads", "reload" in state.load_status, state.load_status)


PANELS = MappingProxyType(
    {
        "environment": environment_controls,
        "sampling": sampling_controls,
        "prompt_playback": prompt_playback,
        "avatar": avatar_controls,
        "model": model_controls,
    }
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    parser.add_argument(
        "--ckpt", default="artifacts/checkpoints/generator/generator_mamba_bs8_e60.pt"
    )
    parser.add_argument(
        "--tokenizer_ckpt", default="artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
    )
    parser.add_argument("--studio_config", default="configs/studio.yml")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--panels", nargs="*", default=list(PANELS))
    parser.add_argument("--out", default="artifacts/outputs/qa/studio_controls")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--pace", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = WidgetTracker()
    config = load_dataclass(StudioConfig, args.studio_config)
    client = motion_service_client(
        args.config,
        args.ckpt,
        args.tokenizer_ckpt,
        "mamba",
        "127.0.0.1",
        args.port,
        out_dir / "motion_service.log",
        service_config="configs/service.yml",
        policy=service_policy(load_service_config("configs/service.yml")),
    )
    viewer_args = argparse.Namespace(
        config=args.config,
        ckpt=args.ckpt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        backbone="mamba",
        temperature=1.0,
        top_p=0.9,
        cfg_scale=6.0,
        steps=0,
        seconds=0.0,
        fps=20,
    )
    viewer = build_studio_viewer(
        args.visible,
        client=client,
        model_dir=str(config.fit.model_dir),
        device=config.fit.device,
        args=viewer_args,
        config=config,
        reload_config=lambda: load_dataclass(StudioConfig, args.studio_config),
    )
    driver = HumanDriver(viewer, out_dir, pace=args.pace)
    pointer = Pointer(driver, tracker)
    driver.start()
    driver.wait(1.0)
    for name in args.panels:
        print(f"== {name}", flush=True)
        try:
            PANELS[name](driver, pointer)
        except Exception as exc:
            traceback.print_exc()
            driver.check("panel test runs to completion", False, f"{type(exc).__name__}: {exc}")
        driver.screenshot("end")
    MotionStreamProtocol.send_json(client.io, {"cmd": "shutdown"})
    tracker.restore()
    failed = [f for f in driver.findings if not f.passed]
    report = {
        "checks": len(driver.findings),
        "failed": len(failed),
        "findings": [f.__dict__ for f in driver.findings],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{len(driver.findings) - len(failed)}/{len(driver.findings)} control checks passed")
    for finding in failed:
        print(f"  FAIL {finding.scenario}: {finding.check} -- {finding.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
