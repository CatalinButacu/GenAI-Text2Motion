from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import imgui
import numpy as np
from aitviewer.utils.pyqt6_window import Keys as StudioWindowKeys
from moderngl_window.context.headless.window import Window as HeadlessWindow

from text2motion.app.schema import load_dataclass
from text2motion.streaming.protocol import MotionStreamProtocol, connect_or_start_motion_server
from text2motion.studio.config import StudioConfig
from text2motion.studio.viewer import StreamingStudioViewer

UNIQUE_NODE_NAMES = ("Motion (streaming)", "Avatar (SMPL-X)", "Avatar (rest)")
DISPLAY_REFRESH_HZ = 60.0


class HeadlessStudio(StreamingStudioViewer):
    window_type = "headless"


class VisibleStudio(StreamingStudioViewer):
    pass


def build_studio_viewer(visible: bool, **kwargs) -> StreamingStudioViewer:
    studio_class = VisibleStudio if visible else HeadlessStudio
    return studio_class(**kwargs)


HeadlessWindow.keys = StudioWindowKeys


@dataclass
class Finding:
    scenario: str
    check: str
    passed: bool
    detail: str


@dataclass
class HumanDriver:
    viewer: HeadlessStudio
    out_dir: Path
    scenario: str = "setup"
    findings: list[Finding] = field(default_factory=list)
    frames: int = 0
    reported_invariants: set = field(default_factory=set)
    pace: float = 1.0

    def start(self) -> None:
        if self.viewer.window_type != "headless":
            imgui.get_io().mouse_draw_cursor = True
        self.viewer._init_scene()
        self.viewer.timer.start()
        self.viewer._last_frame_rendered_at = self.viewer.timer.time

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.findings.append(Finding(self.scenario, name, bool(passed), detail))
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {self.scenario}: {name}{' -- ' + detail if detail else ''}", flush=True)
        return passed

    def invariant(self, name: str, passed: bool, detail: str) -> None:
        key = (self.scenario, name)
        if passed or key in self.reported_invariants:
            return
        self.reported_invariants.add(key)
        self.check(f"invariant: {name}", False, f"frame {self.frames}: {detail}")

    def frame(self) -> None:
        frame_started = time.perf_counter()
        current_time, delta = self.viewer.timer.next_frame()
        try:
            self.viewer.window.clear()
            self.viewer.window.render(current_time, delta)
            self.viewer.window.swap_buffers()
        except Exception as exc:
            traceback.print_exc()
            self.invariant("render never raises", False, f"{type(exc).__name__}: {exc}")
        self.frames += 1
        self.check_invariants()
        remaining = 1.0 / DISPLAY_REFRESH_HZ - (time.perf_counter() - frame_started)
        if remaining > 0:
            time.sleep(remaining)

    def check_invariants(self) -> None:
        state = self.viewer.state
        names = Counter(node.name for node in self.viewer.scene.nodes)
        for unique in UNIQUE_NODE_NAMES:
            self.invariant(
                f"at most one '{unique}' node", names[unique] <= 1, f"{names[unique]} in scene"
            )
        motion = state.nodes.motion
        body_frames = state.stream_buffers.body_frames
        if motion is not None and state.nodes.body is not None:
            skeleton_frames = len(motion.joint_positions)
            self.invariant(
                "body never has more frames than the skeleton",
                body_frames <= skeleton_frames,
                f"body {body_frames} > skeleton {skeleton_frames}",
            )
        n_frames = self.viewer.scene.n_frames
        self.invariant(
            "playhead inside the timeline",
            0 <= self.viewer.scene.current_frame_id < max(1, n_frames),
            f"playhead {self.viewer.scene.current_frame_id} of {n_frames}",
        )
        lowered = state.status.lower()
        self.invariant(
            "status never reports an error",
            "error" not in lowered and "failed" not in lowered,
            state.status,
        )
        if motion is not None:
            finite = np.isfinite(motion.joint_positions).all()
            self.invariant("skeleton joints are finite", finite, "NaN/inf joints on screen")

    def wait(self, seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            self.frame()

    def wait_until(self, condition, timeout: float) -> bool:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if condition():
                return True
            self.frame()
        return condition()

    def type_text(self, text: str, chars_per_second: float = 14.0) -> None:
        for character in text:
            self.viewer.state.prompt_text += character
            self.wait(self.pace / chars_per_second)

    def linger(self) -> None:
        if self.pace > 1.0:
            self.wait(0.3 * (self.pace - 1.0))

    def clear_prompt(self) -> None:
        self.viewer.state.prompt_text = ""
        self.frame()

    def press_enter(self) -> None:
        self.viewer.generation.submit_prompt()
        self.frame()

    def press_stop(self) -> None:
        self.viewer.generation.stop()
        self.frame()

    def press_key(self, key_name: str) -> None:
        keys = self.viewer.wnd.keys
        self.viewer.key_event(getattr(keys, key_name), keys.ACTION_PRESS, None)
        self.frame()

    def screenshot(self, name: str) -> dict:
        self.frame()
        image = self.viewer.get_current_frame_as_image()
        path = self.out_dir / f"{self.scenario}_{name}.png"
        image.save(path)
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        visible_ids = set()
        try:
            visible_ids = set(np.unique(self.viewer.get_current_mask_ids()).tolist())
        except Exception as exc:
            print(f"  [warn] mask ids unavailable: {type(exc).__name__}: {exc}")
        return {"path": str(path), "pixel_std": float(pixels.std()), "visible_ids": visible_ids}

    def node_visible(self, shot: dict, node) -> bool:
        if node is None:
            return False
        uids = {node.uid}
        pending = list(node.nodes)
        while pending:
            child = pending.pop()
            uids.add(child.uid)
            pending.extend(child.nodes)
        return bool(uids & shot["visible_ids"])


def scenario_type_generate_watch(driver: HumanDriver) -> None:
    state = driver.viewer.state
    driver.scenario = "type_generate_watch"
    state.steps = 40
    driver.clear_prompt()
    driver.type_text("a person walks forward")
    driver.press_enter()
    driver.check("pressing Enter starts a generation", state.generating, state.status)
    driver.check(
        "first motion reaches the screen within 3 s",
        driver.wait_until(lambda: state.nodes.motion is not None, 3.0),
        state.status,
    )
    finished = driver.wait_until(lambda: not state.generating, 30.0)
    driver.check("a 40-step generation finishes within 30 s", finished, state.status)
    driver.check(
        "the body mesh is fitted on every frame",
        driver.wait_until(
            lambda: (
                state.nodes.motion is not None
                and state.stream_buffers.body_frames == len(state.nodes.motion.joint_positions)
            ),
            20.0,
        ),
        f"body {state.stream_buffers.body_frames} frames",
    )
    driver.wait(1.0)
    shot = driver.screenshot("after_generation")
    driver.check(
        "the frame is not blank", shot["pixel_std"] > 5.0, f"pixel std {shot['pixel_std']:.1f}"
    )
    driver.check(
        "the body mesh is actually drawn on screen",
        driver.node_visible(shot, state.nodes.body),
        shot["path"],
    )
    driver.check(
        "the skeleton is hidden while the mesh covers the frame",
        not driver.node_visible(shot, state.nodes.motion),
        shot["path"],
    )
    driver.check("the playhead keeps moving after generation", driver.viewer.run_animations, "")


def scenario_double_enter(driver: HumanDriver) -> None:
    state = driver.viewer.state
    driver.scenario = "double_enter"
    state.steps = 20
    driver.clear_prompt()
    driver.type_text("a person jumps")
    driver.press_enter()
    take = state.stream_buffers.take
    driver.press_enter()
    driver.check(
        "a second Enter while generating starts nothing new", state.stream_buffers.take == take, ""
    )
    driver.wait_until(lambda: not state.generating, 30.0)


def scenario_forever_then_stop(driver: HumanDriver) -> None:
    state = driver.viewer.state
    driver.scenario = "forever_then_stop"
    state.steps = 0
    driver.clear_prompt()
    driver.type_text("a person runs in place")
    driver.press_enter()
    driver.wait(6.0)
    before = 0 if state.nodes.motion is None else len(state.nodes.motion.joint_positions)
    driver.check("forever mode streams past 6 s", before >= 80, f"{before} frames on screen")
    driver.press_stop()
    stopped = driver.wait_until(lambda: not state.generating, 5.0)
    driver.check("Stop ends the generation within 5 s", stopped, state.status)
    frames_at_stop = 0 if state.nodes.motion is None else len(state.nodes.motion.joint_positions)
    driver.wait(2.0)
    after = 0 if state.nodes.motion is None else len(state.nodes.motion.joint_positions)
    driver.check(
        "no frames arrive after Stop", after == frames_at_stop, f"{frames_at_stop} -> {after}"
    )
    state.steps = 20
    driver.clear_prompt()
    driver.type_text("a person sits down")
    driver.press_enter()
    driver.check(
        "a new prompt right after Stop generates normally",
        driver.wait_until(lambda: not state.generating, 30.0)
        and "error" not in state.status.lower(),
        state.status,
    )
    expected = 20 * state.downsample * state.upsample
    driver.wait_until(
        lambda: (
            state.nodes.motion is not None and len(state.nodes.motion.joint_positions) >= expected
        ),
        2.0,
    )
    new_take = 0 if state.nodes.motion is None else len(state.nodes.motion.joint_positions)
    driver.check(
        "the new take replaces the old one on the timeline",
        new_take == expected,
        f"{new_take} skeleton frames for a 20-step take",
    )
    driver.wait_until(lambda: state.stream_buffers.body_frames == new_take, 20.0)
    shot = driver.screenshot("new_take_after_stop")
    driver.check(
        "the new take's mesh is drawn",
        driver.node_visible(shot, state.nodes.body),
        f"body {state.stream_buffers.body_frames} frames, {shot['path']}",
    )


def scenario_keyboard(driver: HumanDriver) -> None:
    state = driver.viewer.state
    driver.scenario = "keyboard"
    skin = state.show_skin
    shadows = driver.viewer.shadows_enabled
    driver.press_key("H")
    driver.check(
        "H toggles the skin", state.show_skin != skin, f"show_skin {skin} -> {state.show_skin}"
    )
    driver.check(
        "the skin key leaves the shadows alone",
        driver.viewer.shadows_enabled == shadows,
        f"shadows {shadows} -> {driver.viewer.shadows_enabled}",
    )
    shot = driver.screenshot("skin_hidden")
    driver.check(
        "with skin hidden the skeleton is drawn",
        driver.node_visible(shot, state.nodes.motion),
        shot["path"],
    )
    driver.press_key("H")
    skin = state.show_skin
    shadows = driver.viewer.shadows_enabled
    driver.press_key("S")
    driver.check(
        "S toggles only the shadows",
        driver.viewer.shadows_enabled != shadows and state.show_skin == skin,
        f"shadows {shadows} -> {driver.viewer.shadows_enabled}, skin {skin} -> {state.show_skin}",
    )
    driver.press_key("S")
    driver.press_key("R")
    driver.check("R reloads settings", "reload" in state.load_status, state.load_status)


def scenario_live_typing(driver: HumanDriver) -> None:
    state = driver.viewer.state
    driver.scenario = "live_typing"
    state.steps = 24
    driver.clear_prompt()
    state.live.enabled = True
    driver.type_text("a person walks")
    driver.check(
        "live mode starts generating after a typing pause",
        driver.wait_until(lambda: state.generating, 3.0),
        state.status,
    )
    driver.type_text(" backwards")
    driver.check(
        "editing the prompt regenerates for the new text",
        driver.wait_until(lambda: state.live.last_dispatched == "a person walks backwards", 5.0),
        f"last dispatched {state.live.last_dispatched!r}",
    )
    driver.check(
        "live generation settles",
        driver.wait_until(lambda: not state.generating and not state.live.regen_pending, 30.0),
        state.status,
    )
    state.live.enabled = False
    driver.frame()


def scenario_switch_models(driver: HumanDriver) -> None:
    state = driver.viewer.state
    driver.scenario = "switch_models"
    labels = [entry.key for entry in state.models]
    transformer = next(
        (i for i, key in enumerate(labels) if key.startswith("transformer:") and "thesis" in key),
        None,
    )
    mamba = next(
        (i for i, key in enumerate(labels) if key.startswith("mamba:") and "thesis" in key), None
    )
    if not driver.check(
        "the registry lists both thesis models",
        transformer is not None and mamba is not None,
        str(labels),
    ):
        return
    state.steps = 0
    state.model_idx = transformer
    driver.viewer.generation.start_model_load()
    loaded = driver.wait_until(lambda: not state.loading_model, 120.0)
    driver.check(
        "the transformer loads", loaded and "active" in state.load_status, state.load_status
    )
    driver.check(
        "a re-anchored transformer keeps 'run forever' available",
        state.max_steps == 0 and state.steps == 0,
        f"max_steps {state.max_steps} steps {state.steps}",
    )
    state.steps = 20
    driver.clear_prompt()
    driver.type_text("a person waves")
    driver.press_enter()
    driver.check(
        "the transformer generates",
        driver.wait_until(lambda: not state.generating, 30.0) and state.nodes.motion is not None,
        state.status,
    )
    state.model_idx = mamba
    driver.viewer.generation.start_model_load()
    loaded = driver.wait_until(lambda: not state.loading_model, 120.0)
    driver.check(
        "switching back to Mamba loads", loaded and "active" in state.load_status, state.load_status
    )


def scenario_long_forever_stream(driver: HumanDriver) -> None:
    state = driver.viewer.state
    viewer = driver.viewer
    driver.scenario = "long_forever_stream"
    state.steps = 0
    driver.clear_prompt()
    driver.type_text("walk slowly then jump and walk back")
    driver.press_enter()
    driver.wait(3.0)
    samples = []
    for _ in range(6):
        motion = state.nodes.motion
        frame = viewer.scene.current_frame_id
        pose = (
            None
            if motion is None
            else np.array(motion.joint_positions[min(frame, len(motion.joint_positions) - 1)])
        )
        samples.append(
            (
                time.perf_counter(),
                state.stream_buffers.frames_emitted,
                state.stream_buffers.trimmed_frames + frame,
                pose,
                driver.node_visible(driver.screenshot("forever_sample"), state.nodes.body),
            )
        )
        driver.wait(4.0)
    elapsed = samples[-1][0] - samples[0][0]
    emitted_rate = (samples[-1][1] - samples[0][1]) / elapsed
    played_rate = (samples[-1][2] - samples[0][2]) / elapsed
    driver.check(
        "a forever stream is paced to playback, not flooded",
        emitted_rate < 2.0 * viewer.playback_fps,
        f"{emitted_rate:.0f} frames/s generated vs {viewer.playback_fps:.0f} fps playback",
    )
    driver.check(
        "the playhead keeps moving through new motion",
        played_rate > 0.5 * viewer.playback_fps,
        f"{played_rate:.1f} frames/s played",
    )
    poses = [sample[3] for sample in samples if sample[3] is not None]
    moved = [float(np.abs(a - b).max()) for a, b in zip(poses, poses[1:])]
    driver.check(
        "the avatar pose changes over time",
        len(moved) >= 3 and min(moved) > 0.01,
        f"max joint change between samples (m): {[round(m, 3) for m in moved]}",
    )
    visible = [sample[4] for sample in samples]
    driver.check(
        "the mesh is on screen while streaming",
        sum(visible) >= len(visible) - 1,
        f"mesh visible in {sum(visible)}/{len(visible)} samples",
    )
    driver.press_stop()
    driver.check(
        "Stop ends the long stream",
        driver.wait_until(lambda: not state.generating, 5.0),
        state.status,
    )
    labels = [entry.key for entry in state.models]
    transformer = next(
        i for i, key in enumerate(labels) if key.startswith("transformer:") and "thesis" in key
    )
    mamba = next(i for i, key in enumerate(labels) if key.startswith("mamba:") and "thesis" in key)
    for index in (transformer, mamba):
        state.model_idx = index
        driver.viewer.generation.start_model_load()
        driver.check(
            f"Load model after a long stream works ({labels[index]})",
            driver.wait_until(lambda: not state.loading_model, 60.0)
            and "active" in state.load_status,
            state.load_status,
        )
    state.steps = 20
    driver.clear_prompt()
    driver.type_text("a person waves")
    driver.press_enter()
    driver.check(
        "generation works after the long stream and model switches",
        driver.wait_until(lambda: not state.generating, 30.0) and state.nodes.motion is not None,
        state.status,
    )


SCENARIOS = {
    "type_generate_watch": scenario_type_generate_watch,
    "double_enter": scenario_double_enter,
    "forever_then_stop": scenario_forever_then_stop,
    "keyboard": scenario_keyboard,
    "live_typing": scenario_live_typing,
    "switch_models": scenario_switch_models,
    "long_forever_stream": scenario_long_forever_stream,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    parser.add_argument(
        "--ckpt", default="artifacts/checkpoints/generator/generator_mamba_bs8_e60.pt"
    )
    parser.add_argument(
        "--tokenizer_ckpt", default="artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt"
    )
    parser.add_argument("--backbone", default="mamba")
    parser.add_argument("--studio_config", default="configs/studio.yml")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--out", default="artifacts/outputs/qa/studio")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--pace", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_dataclass(StudioConfig, args.studio_config)
    client = connect_or_start_motion_server(
        args.config,
        args.ckpt,
        args.tokenizer_ckpt,
        args.backbone,
        "127.0.0.1",
        args.port,
        out_dir / "motion_service.log",
        service_config="configs/service.yml",
    )
    viewer_args = argparse.Namespace(
        config=args.config,
        ckpt=args.ckpt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        backbone=args.backbone,
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
    driver.start()
    driver.wait(1.0)
    for name in args.scenarios:
        print(f"== {name}", flush=True)
        try:
            SCENARIOS[name](driver)
        except Exception as exc:
            traceback.print_exc()
            driver.check("scenario runs to completion", False, f"{type(exc).__name__}: {exc}")
    MotionStreamProtocol.send_json(client.io, {"cmd": "shutdown"})
    failed = [f for f in driver.findings if not f.passed]
    report = {
        "frames_rendered": driver.frames,
        "checks": len(driver.findings),
        "failed": len(failed),
        "findings": [f.__dict__ for f in driver.findings],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\n{len(driver.findings) - len(failed)}/{len(driver.findings)} checks passed, {driver.frames} frames rendered"
    )
    for finding in failed:
        print(f"  FAIL {finding.scenario}: {finding.check} -- {finding.detail}")
    print(f"report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
